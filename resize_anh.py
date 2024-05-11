from PIL import Image

def resize_image(input_path, output_path, new_width, new_height):
    # Mở ảnh từ đường dẫn đầu vào
    original_image = Image.open(input_path)

    # Resize ảnh
    resized_image = original_image.resize((new_width, new_height))

    # Lưu ảnh sau khi resize vào đường dẫn đầu ra
    resized_image.save(output_path)

# Thay đổi các giá trị sau đây theo nhu cầu của bạn
input_image_path = 'img/IMG_7152.jpg'
output_image_path = 'anh/82.png'
new_width = 350  # Độ rộng mới của ảnh
new_height = 500  # Độ cao mới của ảnh

# Gọi hàm để resize ảnh
resize_image(input_image_path, output_image_path, new_width, new_height)
