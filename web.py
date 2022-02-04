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
import json

from scripts.align_faces_parallel import align_face
from scripts.run_domain_adaptation import test, DomainAdaptation
from options.test_options import TestOptions
import dlib
from helpers import dotdict, current_milli_time, lock, Thread
import asyncio

from U2Net import U2NETHelper

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
    
    html_script = """
        const form = document.getElementById('form');
        const message = document.getElementById('message');
        const response = document.getElementById('response');


        $('#form').submit(function(e) {

            form.style.display = 'none';
            message.style.display = '';
            response.innerHTML = '';

            $.ajax({
                url: '/process',
                timeout: 3600000,
                type: 'POST',
                data: new FormData(this),
                processData: false,
                contentType: false,
                cache: false,
                success: function(data) {
                    const obj = JSON.parse(data);

                    form.style.display = '';
                    message.style.display = 'none';

                    if (obj.hasOwnProperty('processed')) {
                        var res =
                            '<table style="width: 100%">' +
                            '<tr>' +
                            '<td><img style="width:512px" src="' + obj.orig + '"></td>' +
                            '<td><img style="width:512px" src="' + obj.processed + '"></td>' +
                            '</tr>' +
                            '</table>';

                        response.innerHTML = res;
                    } else {
                        var res = '';
                        for (const [key, value] of Object.entries(obj)) {
                            res += '<div style="display: block; float: left; width: 256px; height: 300px;">' + key + '<br/><img style="width:256px" src="' + value + '"></div>'
                        }
                        response.innerHTML = res;
                    }

                },
            });

            e.preventDefault();
        });
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

            Use all styles: <input type="checkbox" name="test_all">
            <br/><br/>

            Remove background: <input type="checkbox" name="remove_bg">
            <br/><br/>

            <input type="submit" name="submit" value="Submit"/> 
        </form>
        <div id="response"></div>
        </body>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
       {html_script}
        
        </script>
    """)


completed = True
def inference(pretrained_model_path: str = "", remove_bg: bool = False):
    global completed
    domainAdaptation.process(pretrained_model_path)
    if remove_bg:
        u2net.test("experiment/domain_adaptation_results/test.png", "experiment/domain_adaptation_results/test.png")
    completed = True

processing = False

@app.post("/process")
async def process(request: Request, photo: UploadFile = File(...), checkpoint: str = Form(""), test_all : bool = Form(False), remove_bg : bool = Form(False), response_class=HTMLResponse):
    global settings, workers, tasks
    global completed, processing

    if photo != None:
        while processing == True:
            await asyncio.sleep(2)
        processing = True
        
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
                processing = False
                return HTMLResponse(json.dumps({"error": "Face not found"}))
            
            photo_path = os.path.join("test_data", "aligned", "test.png")
            face_img.save(photo_path) # save aligned photo

            #for f in glob.glob('./test/*'):
            #    os.remove(f)

            '''
            '''

            print("aligned")

            if test_all:
                copyfile(photo_path, os.path.join("out", filename + "_orig.png"))

                results = {}
                results["orig"] = os.path.join("./", "out", filename + "_orig.png")

                for pretrained_model_path in sorted(glob.glob(os.path.join(pretrained_models_path, "*.pt"))):
                    
                    completed = False
                    Thread(inference, pretrained_model_path, remove_bg)
                    while completed == False:
                        await asyncio.sleep(2)

                    name = os.path.basename(pretrained_model_path)

                    copyfile("experiment/domain_adaptation_results/test.png", os.path.join("out", filename + "_" + name + "_processed.png"))
                    
                    
                    results[name] = os.path.join("./", "out", filename + "_" + name + "_processed.png")
                    
                    print("finished", name)

                print("finished all")
                print(results)

                processing = False
                return(HTMLResponse(json.dumps(results)))
                
                return(HTMLResponse(f"""
                    <html><body>
                    <div style="display: block; float: left; width: 256px; height: 300px;">&nbsp;<br/><img style="width:256px" src="{os.path.join("./", "out", filename + "_orig.png")}"></div>
                    {"".join(html)}
                    </body></html>
                """))
                
            else:

                completed = False
                Thread(inference, os.path.join(pretrained_models_path, checkpoint), remove_bg)
                while completed == False:
                    await asyncio.sleep(2)

                print("finished")

                copyfile(photo_path, os.path.join("out", filename + "_orig.png"))
                copyfile("experiment/domain_adaptation_results/test.png", os.path.join("out", filename + "_processed.png"))

                processing = False
                return HTMLResponse(json.dumps({
                    "orig" : os.path.join("./", "out", filename + "_orig.png"),
                    "processed" : os.path.join("./", "out", filename + "_processed.png"),
                }))

        except Exception as e:
            processing = False
            return HTMLResponse(json.dumps({"error": "Image is not available or corrupted. " + str(e)}))

    return HTMLResponse("")

def server():
    uvicorn.run(app, host=localhost, port=8080)


domainAdaptation = None

u2net = None
 

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

    u2net = U2NETHelper.U2NETHelper()

    domainAdaptation = DomainAdaptation(opt)

    server()

# python web.py