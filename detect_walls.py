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

    return floorplan_buf, room_buf.getvalue(), icon_buf.getvalue(), mesh_str


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: python detect_walls.py [image_file]")
        quit()

    img_path = sys.argv[1]
    parse_floorplan(img_path)
