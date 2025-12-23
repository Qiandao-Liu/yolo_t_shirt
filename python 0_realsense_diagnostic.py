"""
RealSense 相机诊断工具
用于检测和排查 D435 连接问题
"""

import sys

def check_realsense():
    print("=" * 50)
    print("RealSense 相机诊断工具")
    print("=" * 50)
    
    # 1. 检查 pyrealsense2 是否安装
    print("\n[1] 检查 pyrealsense2 模块...")
    try:
        import pyrealsense2 as rs
        print(f"    ✓ pyrealsense2 版本: {rs.__version__}")
    except ImportError as e:
        print(f"    ✗ pyrealsense2 未安装!")
        print(f"    请运行: pip install pyrealsense2")
        return False
    
    # 2. 检查连接的设备
    print("\n[2] 扫描已连接的 RealSense 设备...")
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("    ✗ 未检测到任何 RealSense 设备!")
        print("\n    可能的原因:")
        print("    - USB 线缆未正确连接")
        print("    - 需要使用 USB 3.0 端口 (蓝色)")
        print("    - USB 线缆不支持数据传输 (仅充电线)")
        print("    - 需要安装 Intel RealSense SDK")
        print("    - Linux 用户可能需要设置 udev 规则")
        return False
    
    print(f"    ✓ 检测到 {len(devices)} 个设备:")
    
    for i, dev in enumerate(devices):
        print(f"\n    设备 {i + 1}:")
        print(f"      名称: {dev.get_info(rs.camera_info.name)}")
        print(f"      序列号: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"      固件版本: {dev.get_info(rs.camera_info.firmware_version)}")
        print(f"      USB 类型: {dev.get_info(rs.camera_info.usb_type_descriptor)}")
        
        # 检查 USB 版本
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
        if "2." in usb_type:
            print(f"      ⚠ 警告: 使用 USB 2.0 连接，建议使用 USB 3.0 以获得最佳性能")
    
    # 3. 尝试获取可用的流配置
    print("\n[3] 检查可用的流配置...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 获取第一个设备
    device = devices[0]
    serial = device.get_info(rs.camera_info.serial_number)
    config.enable_device(serial)
    
    # 尝试不同的分辨率
    resolutions = [
        (1280, 720, 30),
        (1280, 720, 15),
        (848, 480, 30),
        (640, 480, 30),
        (640, 480, 15),
        (424, 240, 30),
    ]
    
    working_config = None
    
    for width, height, fps in resolutions:
        try:
            test_config = rs.config()
            test_config.enable_device(serial)
            test_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            
            # 尝试解析配置
            pipeline_profile = test_config.resolve(rs.pipeline_wrapper(pipeline))
            print(f"    ✓ 支持: {width}x{height} @ {fps}fps (RGB)")
            
            if working_config is None:
                working_config = (width, height, fps)
                
        except Exception as e:
            print(f"    ✗ 不支持: {width}x{height} @ {fps}fps")
    
    # 4. 尝试实际启动相机
    print("\n[4] 尝试启动相机流...")
    
    if working_config:
        width, height, fps = working_config
        try:
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            
            pipeline_profile = pipeline.start(config)
            print(f"    ✓ 相机成功启动! ({width}x{height} @ {fps}fps)")
            
            # 获取几帧测试
            print("\n[5] 测试获取帧...")
            for i in range(5):
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    print(f"    ✓ 成功获取第 {i+1} 帧")
                    
            pipeline.stop()
            print("\n" + "=" * 50)
            print("诊断完成: 相机工作正常!")
            print("=" * 50)
            
            print(f"\n建议的配置: {width}x{height} @ {fps}fps")
            print("你可以在录制程序中使用此配置")
            
            return True
            
        except Exception as e:
            print(f"    ✗ 启动失败: {e}")
            pipeline.stop() if pipeline else None
            
    else:
        print("    ✗ 未找到可用的流配置")
    
    print("\n" + "=" * 50)
    print("诊断完成: 存在问题，请查看上述信息")
    print("=" * 50)
    
    return False


def print_troubleshooting():
    print("\n" + "=" * 50)
    print("故障排除建议")
    print("=" * 50)
    
    print("""
1. 检查 USB 连接:
   - 使用 USB 3.0 端口 (通常是蓝色)
   - 确保使用原装或高质量 USB 3.0 数据线
   - 尝试不同的 USB 端口
   - 避免使用 USB 集线器

2. Windows 用户:
   - 安装 Intel RealSense SDK: https://github.com/IntelRealSense/librealsense/releases
   - 在设备管理器中检查是否识别到设备
   - 更新 USB 控制器驱动

3. Linux 用户:
   - 安装 udev 规则:
     sudo cp ~/.local/lib/python*/site-packages/pyrealsense2/udev/99-realsense-libusb.rules /etc/udev/rules.d/
     sudo udevadm control --reload-rules && sudo udevadm trigger
   
   - 或手动创建规则:
     echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", MODE="0666"' | sudo tee /etc/udev/rules.d/99-realsense.rules
     sudo udevadm control --reload-rules && sudo udevadm trigger

4. macOS 用户:
   - RealSense SDK 对 macOS 支持有限
   - 建议使用 Linux 或 Windows

5. 重启相机:
   - 拔掉 USB 线等待 5 秒后重新插入
   - 重启电脑后再试

6. 检查其他程序:
   - 确保没有其他程序 (如 RealSense Viewer) 正在使用相机
   - 关闭所有可能使用相机的应用
""")


if __name__ == "__main__":
    success = check_realsense()
    
    if not success:
        print_troubleshooting()
    
    input("\n按 Enter 键退出...") 