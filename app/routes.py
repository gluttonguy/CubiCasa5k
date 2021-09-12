from flask import render_template, flash, redirect, url_for
from flask.wrappers import Request
from flask import render_template, Blueprint, url_for, \
    redirect, flash, request, jsonify, session, send_file
from app import app
import io
import traceback
import base64

import detect_walls

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload_floorplan', methods=['POST'])
def upload_floorplan():
    response={}
    try:
        uploaded_file = request.files.get('file',None)
        if uploaded_file is None:
            return jsonify({'error':'No floorplan image file uploaded'})

        file_bytes = uploaded_file.read()
        file_size = len(file_bytes)
        file_name=uploaded_file.filename

        floorplan_buf,room_buf,icon_buf,mesh_str=detect_walls.parse_floorplan(io.BytesIO(file_bytes))

        response['floorplan_png']=base64.b64encode(floorplan_buf).decode()
        response['rooms_png']=base64.b64encode(room_buf).decode()
        response['icons_png']=base64.b64encode(icon_buf).decode()
        if mesh_str is not None: response['mesh_obj']=mesh_str
        
        return jsonify(response)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error':'Invalid floorplan image file or processing error'})    
    finally:
        pass



