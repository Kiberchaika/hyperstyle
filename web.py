from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi import Request, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
import uvicorn
import json

import time
from PIL import Image, ExifTags
import time
import os
import platform
import glob
from shutil import copyfile
import threading

from scripts.align_faces_parallel import align_face
from scripts.run_domain_adaptation import test, DomainAdaptation
from options.test_options import TestOptions
import dlib
from helpers import dotdict, current_milli_time, lock, Thread

# server app
app = FastAPI()
app.mount("/out", StaticFiles(directory="out"), name="out")

localhost = 'localhost' if platform.system() == 'Windows' else '0.0.0.0'
 
pretrained_models_path = "pretrained_models/exp3"

@app.get("/")
async def test_api(response_class=HTMLResponse):

    checkpoints_html_options = ""
    for file in sorted(glob.glob(os.path.join(pretrained_models_path, "*.pt"))):
        filename = os.path.basename(file)
        checkpoints_html_options += f'<option value="{filename}">{filename}</option>'

    html_header = """
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            html {
                font-size: 20px;
            }
        </style>    
    """
    
    return HTMLResponse(f"""
        {html_header}
        <body>
        <h2>Test API</h2><br/><br/>
        <div id="message" style="display:none;">wait...</div>
        <form id="form" action=\"/process\" method=\"POST\" enctype=\"multipart/form-data\">
            
            Choose photo: <input type='file' id="file" name="photo" accept=".gif,.jpg,.jpeg,.png">
            <br/><br/>
            
            Style: <select name="checkpoint">{checkpoints_html_options}</select>
            <br/><br/>

            <input type="submit" value="Submit"/> 
        </form>
        </body>
        <script>
        const form = document.getElementById('form');
        const message = document.getElementById('message');

        function submit(event) {{
            form.style.display = 'none';
            message.style.display = '';
        }}

        form.addEventListener('submit', submit);
        </script>
    """)

@app.post("/process")
async def process(request: Request, photo: UploadFile = File(...), checkpoint: str = Form(""), response_class=HTMLResponse):
    global settings, workers, tasks

    if photo != None:
        with lock:
            try:
                filename = str(current_milli_time())

                image = Image.open(photo.file._file) 

                # fixed rotation from exif
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation]=='Orientation':
                        break
                
                if image._exif != None:
                    exif = image._getexif()
                    if orientation in exif:
                        if exif[orientation] == 3:
                            image=image.rotate(180, expand=True)
                        elif exif[orientation] == 6:
                            image=image.rotate(270, expand=True)
                        elif exif[orientation] == 8:
                            image=image.rotate(90, expand=True)

                photo_path = os.path.join("test_data", filename + ".png")
 
                # resize with aspect
                fixed_height = 1024
                height_percent = (fixed_height / float(image.size[1]))
                width_size = int((float(image.size[0]) * float(height_percent)))
                face_img = image.resize((width_size, fixed_height))
               
                image.save(photo_path) # save uploaded photo
                
                predictor = dlib.shape_predictor("pretrained_models/shape_predictor_68_face_landmarks.dat")
                face_img = align_face(photo_path, predictor)
                if face_img is None:
                    return HTMLResponse("[ERROR] Face not found")
                
                photo_path = os.path.join("test_data", "aligned", "test.png")
                face_img.save(photo_path) # save aligned photo

                #for f in glob.glob('./test/*'):
                #    os.remove(f)

                '''
                 '''

                print("aligned")

                domainAdaptation.process(os.path.join(pretrained_models_path, checkpoint))

                print("finished")

                copyfile(photo_path, os.path.join("out", filename + "_orig.png"))
                copyfile("experiment/domain_adaptation_results/test.png", os.path.join("out", filename + "_processed.png"))

                return(HTMLResponse(f"""
                    <html><body>
                    <table style="width: 100%">
                    <tr>
                    <td><img style="width:512px" src="{os.path.join("./", "out", filename + "_orig.png")}"></td>
                    <td><img style="width:512px" src="{os.path.join("./", "out", filename + "_processed.png")}"></td>
                    </tr>
                    </table>
                    </body></html>
                """))

            except Exception as e:
                return HTMLResponse("[ERROR] Image is not available or corrupted.<br/> " + str(e))

    return HTMLResponse("")

def server():
    uvicorn.run(app, host=localhost, port=8080)


domainAdaptation = None

if __name__ == '__main__':
    
    opt = TestOptions().parse()  # get test options
    opt.exp_dir = "experiment"
    opt.checkpoint_path = "pretrained_models/hyperstyle_ffhq.pt"
    opt.finetuned_generator_checkpoint_path = ""
    opt.data_path = "test_data/aligned"
    opt.test_batch_size = 1
    opt.test_workers = 0
    opt.n_iters_per_batch = 2 
    opt.load_w_encoder = True
    opt.w_encoder_checkpoint_path = "pretrained_models/faces_w_encoder.pt"
    opt.restyle_n_iterations = 2  

    domainAdaptation = DomainAdaptation(opt)

    server()

# python web.py