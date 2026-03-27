from flask import Flask, render_template, request, send_from_directory, flash
import os
import cv2
import traceback
from realesrgan import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"  
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

upsampler = None
def get_upsampler():
    global upsampler
    if upsampler is None:
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type='prelu'
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path='realesr-general-x4v3.pth',
            model=model,
            tile=64,       
            tile_pad=10,
            pre_pad=0,
            half=False     
        )
    return upsampler

@app.route('/', methods=['GET', 'POST'])
def index():
    input_file = None
    output_file = None

    if request.method == 'POST':
        try:
            file = request.files['image']
            if file:
                filename = secure_filename(file.filename)

                input_path = os.path.join(UPLOAD_FOLDER, filename)
                output_filename = "output_" + filename
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)

                file.save(input_path)

                img = cv2.imread(input_path)
                if img is None:
                    flash("Không thể đọc ảnh. Vui lòng thử lại.")
                    return render_template('index.html')

                h, w = img.shape[:2]
                max_size = 800
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    img = cv2.resize(img, (int(w*scale), int(h*scale)))

                upsampler_model = get_upsampler()
                output, _ = upsampler_model.enhance(img, outscale=4)

                cv2.imwrite(output_path, output)

                # os.remove(input_path)

                input_file = filename
                output_file = output_filename

        except Exception as e:
            print("ERROR:", e)
            traceback.print_exc()
            flash("Có lỗi xảy ra khi xử lý ảnh. Vui lòng thử lại!")

    return render_template(
        'index.html',
        input_file=input_file,
        output_file=output_file
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)