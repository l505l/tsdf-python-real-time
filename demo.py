"""递增式TSDF融合实现,逐帧构建3D场景"""
import time
import cv2
import numpy as np
import open3d as o3d
import fusion
import os
from pathlib import Path
import time

class ReconstructionSystem:
    def __init__(self, save_path, voxel_size=0.02, update_interval=10):
        """初始化重建系统
        
        Args:
            save_path: 结果保存路径
            voxel_size: 体素大小
            update_interval: 显示更新间隔（帧数）
        """
        self.save_path = save_path
        self.voxel_size = voxel_size
        self.update_interval = update_interval
        self.n_processed = 0
        self.vol_initialized = False
        
        # 创建保存目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # 创建可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='3D Reconstruction', width=1280, height=720)
        
        # 设置渲染选项
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1.0
        opt.show_coordinate_frame = True
        opt.light_on = True
        opt.mesh_show_back_face = True
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        
        # 设置默认相机视角
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.5)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
        # 创建空网格
        self.mesh = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.mesh)
        
        self.running = True
    
    def initialize_volume(self, first_depth, first_pose, cam_intr):
        """根据第一帧初始化TSDF体积"""
        vol_bnds = np.zeros((3,2))
        view_frust_pts = fusion.get_view_frustum(first_depth, cam_intr, first_pose)
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
        
        # 添加边界余量
        vol_bnds[:,0] -= 0.5
        vol_bnds[:,1] += 0.5
        
        self.tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=self.voxel_size)
        self.vol_initialized = True
    
    def process_frame(self, color_image, depth_im, cam_pose, cam_intr):
        """处理单帧数据"""
        if not self.vol_initialized:
            self.initialize_volume(depth_im, cam_pose, cam_intr)
        
        # 整合当前帧
        self.tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
        self.n_processed += 1
        
        # 按指定间隔更新显示
        if self.n_processed % self.update_interval == 0:
            self.update_display()
            self.save_result()  # 定期保存最新结果
        
        # 处理可视化事件
        self.vis.poll_events()
        self.vis.update_renderer()
        
        return self.running
    
    def update_display(self):
        """更新3D显示"""
        verts, faces, norms, colors = self.tsdf_vol.get_mesh()
        
        if len(verts) == 0:
            return
            
        # 更新网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(norms)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # 优化网格
        mesh.compute_triangle_normals()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        
        # 更新显示
        self.vis.clear_geometries()
        self.vis.add_geometry(mesh, reset_bounding_box=True)
    
    def save_result(self):
        """保存当前重建结果"""
        verts, faces, norms, colors = self.tsdf_vol.get_mesh()
        fusion.meshwrite(os.path.join(self.save_path, "mesh_latest.ply"),
                        verts, faces, norms, colors)
    
    def close(self):
        """关闭重建系统"""
        self.save_result()
        # 移除destroy_window调用，让主程序来控制窗口关闭

def wait_for_new_frame(data_path, processed_frames, timeout=3):
    """等待新的数据帧
    
    Args:
        data_path: 数据文件夹路径
        processed_frames: 已处理帧的集合
        timeout: 等待超时时间(秒)
    
    Returns:
        tuple: (color_file, depth_file, pose_file, frame_num) 或 None
    """
    start_time = time.time()
    last_file_time = time.time()  # 记录上次发现新文件的时间
    
    while True:
        files = sorted(Path(data_path).glob("frame-*.color.jpg"))
        for color_file in files:
            frame_num = color_file.name[6:12]  # 提取帧号
            if frame_num in processed_frames:
                continue
                
            depth_file = color_file.parent / f"frame-{frame_num}.depth.png"
            pose_file = color_file.parent / f"frame-{frame_num}.pose.txt"
            
            if depth_file.exists() and pose_file.exists():
                last_file_time = time.time()  # 更新最后发现文件的时间
                return str(color_file), str(depth_file), str(pose_file), frame_num
        
        # 检查是否超时
        if time.time() - last_file_time > timeout:
            print(f"\n{timeout}秒内未检测到新数据，结束处理")
            return None
            
        # 检查是否需要退出
        if os.path.exists(os.path.join(data_path, "STOP")):
            print("\n检测到停止信号，结束处理")
            return None
            
        time.sleep(0.1)  # 短暂休眠避免过度占用CPU

# 使用示例:
if __name__ == "__main__":
    reconstruction = ReconstructionSystem(
        save_path="results",
        voxel_size=0.01,
        update_interval=50
    )
    
    # 设置数据路径
    data_path = "data"
    
    try:
        # 加载相机内参
        camera_intrinsics = np.loadtxt(os.path.join(data_path, "camera-intrinsics.txt"), delimiter=' ')
        
        # 记录已处理的帧
        processed_frames = set()
        
        print("等待数据输入...")
        
        while True:
            # 等待新的数据帧，设置3秒超时
            result = wait_for_new_frame(data_path, processed_frames, timeout=3)
            if result is None:
                print("\n重建完成！请查看结果并手动关闭窗口...")
                # 保持窗口显示直到用户关闭
                while True:
                    try:
                        if not reconstruction.vis.poll_events():
                            break
                        reconstruction.vis.update_renderer()
                        time.sleep(0.1)
                    except KeyboardInterrupt:
                        break
                break  # 退出主处理循环
                
            color_file, depth_file, pose_file, frame_num = result
            
            print(f"处理新帧: {frame_num}")
            
            # 读取并预处理数据
            color_image = cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)
            depth_image = cv2.imread(depth_file, -1).astype(float)
            depth_image /= 1000.  # 转换为米
            depth_image[depth_image == 65.535] = 0  # 处理无效深度值
            camera_pose = np.loadtxt(pose_file)  # 4x4 刚体变换矩阵
            
            # 处理当前帧
            if not reconstruction.process_frame(color_image, depth_image, 
                                             camera_pose, camera_intrinsics):
                break
            
            # 记录已处理的帧
            processed_frames.add(frame_num)
                
    except KeyboardInterrupt:
        print("\n用户中断处理")
    finally:
        # 保存最终结果但不关闭窗口
        reconstruction.save_result()
        
        # 如果是由于KeyboardInterrupt退出，保持窗口显示
        if 'reconstruction' in locals():
            try:
                while True:
                    if not reconstruction.vis.poll_events():
                        break
                    reconstruction.vis.update_renderer()
                    time.sleep(0.1)
            except:
                pass
            finally:
                reconstruction.vis.destroy_window()