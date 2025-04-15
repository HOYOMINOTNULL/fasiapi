from PIL import Image

# 创建一个 640x640 的纯黑色图像
img = Image.new('RGB', (640, 640), (0, 0, 0))

# 保存为 PNG 格式
img.save('black_640x640.png', 'PNG')