from flask import Flask, render_template, request, send_from_directory
import os
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model 1 lần duy nhất
def create_upsampler():
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=4,
        act_type='prelu'
    )
    return RealESRGANer(
        scale=4,
        model_path='realesr-general-x4v3.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )

upsampler = create_upsampler()

@app.route('/', methods=['GET', 'POST'])
def index():
    input_file = None
    output_file = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)

            input_path = os.path.join(UPLOAD_FOLDER, filename)
            output_filename = "output_" + filename
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            file.save(input_path)

            # đọc ảnh
            img = cv2.imread(input_path)

            # upscale
            output, _ = upsampler.enhance(img, outscale=4)

            cv2.imwrite(output_path, output)

            input_file = filename
            output_file = output_filename

    return render_template(
        'index.html',
        input_file=input_file,
        output_file=output_file
    )

# route hiển thị ảnh
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)