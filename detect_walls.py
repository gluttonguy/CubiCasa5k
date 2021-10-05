import trimesh.exchange.obj
import trimesh
from shapely.ops import transform as shapely_transform
from shapely.affinity import scale
from shapely import geometry
from torchvision.transforms import ToTensor
from PIL import Image
from mpl_toolkits.axes_grid1 import AxesGrid
from floortrans.post_prosessing import split_prediction, get_polygons, split_validation
import sys
import io
import math

from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
from floortrans.plotting import segmentation_plot, polygons_to_image, draw_junction_from_dict, discrete_cmap
discrete_cmap()


rot = RotateNTurns()
room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room",
                "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience",
                "Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]
n_classes = 44
split = [21, 12, 11]


# Setup Model
model = get_model('hg_furukawa_original', 51)

model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
model.upsample = torch.nn.ConvTranspose2d(
    n_classes, n_classes, kernel_size=4, stride=4)
#checkpoint = torch.load('/content/drive/MyDrive/CubiCasa5k/model_best_val_loss_var.pkl')
checkpoint = torch.load('model_best_val_loss_var.pkl',
                        map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state'])
model.eval()
# model.cuda()
print("Model loaded.")

def polygon_area_signed(x,y): # positive means counter-clockwise
    x=np.array(x)
    y=np.array(y)
    # coordinate shift
    x_ = x - x.mean()
    y_ = y - y.mean()
    # everything else is the same as maxb's code
    # correction = x_[-1] * y_[0] - y_[-1]* x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    #return 0.5*np.abs(main_area + correction)
    return 0.5*main_area

def Faces(edges,embedding):
    """
    edges: is an undirected graph as a set of undirected edges
    embedding: is a combinatorial embedding dictionary. Format: v1:[v2,v3], v2:[v1], v3:[v1] clockwise ordering of neighbors at each vertex.)

    """

    # Establish set of possible edges
    edgeset = set()
    for edge in edges: # edges is an undirected graph as a set of undirected edges
        edge = list(edge)
        edgeset |= set([(edge[0],edge[1]),(edge[1],edge[0])])

    # Storage for face paths
    faces = []
    path  = []
    for edge in edgeset:
        path.append(edge)
        edgeset -= set([edge])
        break  # (Only one iteration)

    # Trace faces
    while (len(edgeset) > 0):
        neighbors = embedding[path[-1][-1]]
        next_node = neighbors[(neighbors.index(path[-1][-2])+1)%(len(neighbors))]
        tup = (path[-1][-1],next_node)
        if tup == path[0]:
            faces.append(path)
            path = []
            for edge in edgeset:
                path.append(edge)
                edgeset -= set([edge])
                break  # (Only one iteration)
        else:
            path.append(tup)
            edgeset -= set([tup])
    if (len(path) != 0): faces.append(path)
    return faces

def takeSecond(elem):
    return elem[1]
'''
def offset_inward(coords,setbacks,signed_area):
    minx=None
    miny=None
    maxx=None
    maxy=None
    for coord in coords:
        x=coord[0]
        y=coord[1]
        if minx is None or x<minx:
            minx=x
        if maxx is None or x>maxx:
            maxx=x
        if miny is None or y<miny:
            miny=y
        if maxy is None or y>maxy:
            maxy=y
    minx-=1
    maxx+=1
    miny-=1
    maxy+=1

    vertices=[]
    vertex_edges=[]
    shifted_edges=[] #x1 y1 x2 y2
    shifted_edges_vertices=[]
    for i in range(len(coords)-1):
        x1=coords[i][0]
        y1=coords[i][1]
        x2=coords[i+1][0]
        y2=coords[i+1][1]
        if i==len(coords)-2:
            x3=coords[1][0]
            y3=coords[1][1]
            next_i=0
        else:
            x3=coords[i+2][0]
            y3=coords[i+2][1]
            next_i=i+1

        # shift edges
        x4,y4,x5,y5=perpen_shift(x1,y1,x2,y2,setbacks[i],signed_area)
        shifted_edges.append([x4,y4,x5,y5])
        shifted_edges_vertices.append([])

    for i in range(len(shifted_edges)):
        x4,y4,x5,y5=shifted_edges[i]
        for j in range(i+1,len(shifted_edges)):
            x6,y6,x7,y7=shifted_edges[j]

            t1=(x4-x6)*(y4-y5)-(y4-y6)*(x4-x5)
            t2=(x6-x7)*(y4-y5)-(y6-y7)*(x4-x5)
            if t2==0: continue
            q=t1/t2

            x=x6+q*(x6-x7)
            y=y6+q*(y6-y7)
            if x<minx or x>maxx or y<miny or y>maxy: continue

            vertex_index=len(vertices)
            vertices.append([x,y])
            vertex_edges.append([i,j])
            shifted_edges_vertices[i].append([vertex_index,(x-x4)*(x5-x4)+(y-y4)*(y5-y4)])
            shifted_edges_vertices[j].append([vertex_index,(x-x6)*(x7-x6)+(y-y6)*(y7-y6)])

    for l in shifted_edges_vertices:
        l.sort(key=takeSecond)

    incident_matrix=[]
    for i in range(len(vertex_edges)):
        x,y=vertices[i]
        e1,e2=vertex_edges[i]
        for j in range(len(shifted_edges_vertices[e1])):
            if shifted_edges_vertices[e1][j][0]==i:
                index1=j
                break
        for j in range(len(shifted_edges_vertices[e2])):
            if shifted_edges_vertices[e2][j][0]==i:
                index2=j
                break
        my_edges=[]
        if index1>0:
            vi=shifted_edges_vertices[e1][index1-1][0]
            p=vertices[vi]
            my_edges.append([vi,math.atan2(p[1]-y,p[0]-x),-1])
        if index1<len(shifted_edges_vertices[e1])-1:
            vi=shifted_edges_vertices[e1][index1+1][0]
            p=vertices[vi]
            my_edges.append([vi,math.atan2(p[1]-y,p[0]-x),1])
        if index2>0:
            vi=shifted_edges_vertices[e2][index2-1][0]
            p=vertices[vi]
            my_edges.append([vi,math.atan2(p[1]-y,p[0]-x),-1])
        if index2<len(shifted_edges_vertices[e2])-1:
            vi=shifted_edges_vertices[e2][index2+1][0]
            p=vertices[vi]
            my_edges.append([vi,math.atan2(p[1]-y,p[0]-x),1])

        my_edges.sort(key=takeSecond,reverse=True)
        incident_matrix.append(my_edges)

    edges=[]
    embedding=[]
    for i in range(len(incident_matrix)):
        l=[]
        for ti,angle,direction in incident_matrix[i]:
            edges.append([i,ti])
            l.append(ti)
        embedding.append(l)

    faces=Faces(edges,embedding)
    cand_paths=[]
    for path in faces:
        path=[x[0] for x in path]
        paths=[]
        # find all simple polygons
        i=0
        while i < len(path):
            v=path[i]
            for j in range(i):
                if path[j]==v:
                    # extract subpath
                    subpath=path[j:i+1]
                    if len(subpath)>=4: paths.append(subpath)
                    path=path[0:j]+path[i:]
                    i=j
                    break
            i+=1

        if len(path)>=3:
            path.append(path[0])
            paths.append(path)

        for p in paths:
            current_direction=None
            is_valid=True
            for i in range(len(p)-1):
                v=p[i]
                next_v=p[i+1]
                my_edges=incident_matrix[v]
                for vi,angle,direction in my_edges:
                    if vi!=next_v: continue
                    if current_direction is None:
                        current_direction=direction
                    elif current_direction!=direction:
                        is_valid=False
                        break
                if not is_valid:
                    break
            if is_valid:
                edge_indices=[]
                for i in range(len(p)-1):
                    e1=vertex_edges[p[i]]
                    e2=vertex_edges[p[i+1]]
                    edge_indices.append(list(set(e1).intersection(e2))[0])
                #return [vertices[i] for i in p],edge_indices
                cand_paths.append([[vertices[i] for i in p],edge_indices])

    min_area=None
    selected_path=None
    for p in cand_paths:
        area=abs(polygon_area_signed([x[0] for x in p[0]],[x[1] for x in p[0]]))
        if min_area is None or area<min_area:
            min_area=area
            selected_path=p
    
    return selected_path
'''

def reflection(x0):
    return lambda x, y: (2*x0 - x, y)


def parse_floorplan(img_path):
    image = Image.open(img_path)
    width, height = image.size
    if width > height:
        newsize = (512, int(height*512/width))
    else:
        newsize = (int(width*512/height), 512)
    image = image.resize(newsize)

    with io.BytesIO() as output:
        image.save(output, format="PNG")
        floorplan_buf = output.getvalue()

    # unsqueeze to add artificial first dimension
    image = ToTensor()(image).unsqueeze(0)
    if image.shape[1]==4:
        image=image[:,0:3,:,:]

    n_rooms = 12
    n_icons = 11

    with torch.no_grad():
        height = image.shape[2]
        width = image.shape[3]
        img_size = (height, width)

        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred_count = len(rotations)
        prediction = torch.zeros([pred_count, n_classes, height, width])
        for i, r in enumerate(rotations):
            forward, back = r
            # We rotate first the image
            rot_image = rot(image, 'tensor', forward)
            pred = model(rot_image)
            # We rotate prediction back
            pred = rot(pred, 'tensor', back)
            # We fix heatmaps
            pred = rot(pred, 'points', back)
            # We make sure the size is correct
            pred = F.interpolate(pred, size=(height, width),
                                 mode='bilinear', align_corners=True)
            # We add the prediction to output
            prediction[i] = pred[0]

    prediction = torch.mean(prediction, 0, True)
    #rooms_label = label_np[0]
    #icons_label = label_np[1]

    rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
    rooms_pred = np.argmax(rooms_pred, axis=0)

    icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
    icons_pred = np.argmax(icons_pred, axis=0)
    '''
    plt.figure(figsize=(12,12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    rseg = ax.imshow(rooms_pred, cmap='rooms', vmin=0, vmax=n_rooms-0.1)
    cbar = plt.colorbar(rseg, ticks=np.arange(n_rooms) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(room_classes, fontsize=20)
    plt.show()

    plt.figure(figsize=(12,12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    iseg = ax.imshow(icons_pred, cmap='icons', vmin=0, vmax=n_icons-0.1)
    cbar = plt.colorbar(iseg, ticks=np.arange(n_icons) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(icon_classes, fontsize=20)
    plt.show()
    '''
    heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons), 0.2, [1, 2])
    #types_indices=[x for x in range(len(types)) if types[x]['type']!='wall' or types[x]['class']==8]
    #types=[types[x] for x in types_indices]
    #polygons=[polygons[x] for x in types_indices]

    pol_room_seg, pol_icon_seg = polygons_to_image(
        polygons, types, room_polygons, room_types, height, width)
    pol_room_seg[pol_icon_seg==2]=0

    plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    rseg = ax.imshow(pol_room_seg, cmap='rooms', vmin=0, vmax=n_rooms-0.1)
    cbar = plt.colorbar(rseg, ticks=np.arange(
        n_rooms) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(room_classes, fontsize=20)
    plt.tight_layout()
    # plt.show()
    room_buf = io.BytesIO()
    plt.savefig(room_buf, format='png')

    plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    iseg = ax.imshow(pol_icon_seg, cmap='icons', vmin=0, vmax=n_icons-0.1)
    cbar = plt.colorbar(iseg, ticks=np.arange(
        n_icons) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(icon_classes, fontsize=20)
    plt.tight_layout()
    # plt.show()
    icon_buf = io.BytesIO()
    plt.savefig(icon_buf, format='png')

    yourList = []
    openings=[geometry.Polygon(polygons[i]) for i,t in enumerate(types) if t['type']=='icon' and (t['class']==2)]
    openings=geometry.MultiPolygon(openings)

    for i in range(len(types)):
        t = types[i]
        poly = polygons[i]
        if t['type'] == 'wall':
            poly = geometry.Polygon(poly)
            if poly.area == 0:
                continue

            final_polygons=[]
            result=poly.difference(openings)
            if isinstance(result, geometry.Polygon):
                if result.area==0: continue
                final_polygons.append(result)
            elif isinstance(result, geometry.MultiPolygon):
                for poly in result:
                    if poly.area>0: final_polygons.append(poly)
            for poly in final_polygons:
                poly = scale(poly, xfact = -1, origin = (0, 0))
                mesh = trimesh.creation.extrude_polygon(
                    poly, 100)
                yourList.append(mesh)

    mesh_str=None
    if len(yourList)>0:
        vertice_list = [mesh.vertices for mesh in yourList]
        faces_list = [mesh.faces for mesh in yourList]
        faces_offset = np.cumsum([v.shape[0] for v in vertice_list])
        faces_offset = np.insert(faces_offset, 0, 0)[:-1]

        vertices = np.vstack(vertice_list)
        faces = np.vstack(
            [face + offset for face, offset in zip(faces_list, faces_offset)])

        merged__meshes = trimesh.Trimesh(vertices, faces)

        with io.BytesIO() as output:
            # merged__meshes.export("aaa.obj")
            mesh_str = trimesh.exchange.obj.export_obj(merged__meshes)

    floors=[]
    for p,t in zip(room_polygons,room_types):
        if p.area==0: continue
        p = scale(p, xfact = -1, origin = (0, 0))
        mesh = trimesh.creation.extrude_polygon(
            p, 2)
        with io.BytesIO() as output:
            f_mesh_str = trimesh.exchange.obj.export_obj(mesh)
            floors.append([f_mesh_str,t['type']])

    return floorplan_buf, room_buf.getvalue(), icon_buf.getvalue(), mesh_str, floors


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: python detect_walls.py [image_file]")
        quit()

    img_path = sys.argv[1]
    parse_floorplan(img_path)
