# https://memo.sugyan.com/entry/20151203/1449137219
import cv2
import numpy as np
import math
from math import sin, cos
from os import path
import sys
import glob

#cascades_dir = path.normpath(path.join(cv2.__file__, '..', '..', '..', '..', 'share', 'OpenCV', 'haarcascades'))
cascades_dir = './'
cascade_f = cv2.CascadeClassifier(path.join(cascades_dir, 'haarcascade_frontalface_alt2.xml'))
cascade_e = cv2.CascadeClassifier(path.join(cascades_dir, 'haarcascade_eye.xml'))
max_size = 720

def mirror_padding(img):
    padding_y = img.shape[0] // 10
    padding_x = img.shape[1] // 10
    img2 = cv2.copyMakeBorder(img, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_REFLECT_101)
    return img2    

def fitting_rotated_image(img, angle):
    height,width = img.shape[:2]
    center = (int(width/2), int(height/2))
    radians = np.deg2rad(angle)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    new_width = int(abs(np.sin(radians)*height) + abs(np.cos(radians)*width))
    new_height = int(abs(np.sin(radians)*width) + abs(np.cos(radians)*height))

    M[0,2] += int((new_width-width)/2)
    M[1,2] += int((new_height-height)/2)

    return cv2.warpAffine(img, M, (new_width, new_height))

def detect(img):
    # resize if learch image
    rows, cols, _ = img.shape
    if max(rows, cols) > max_size:
        l = max(rows, cols)
        img = cv2.resize(img, (int(cols * max_size / l), int(rows * max_size / l)))
    rows, cols, _ = img.shape
    # create gray image for rotate
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hypot = int(math.ceil(math.hypot(rows, cols)))
    frame = np.zeros((hypot, hypot), np.uint8)
    frame[int((hypot - rows) * 0.5):int((hypot + rows) * 0.5), int((hypot - cols) * 0.5):int((hypot + cols) * 0.5)] = gray

    def translate(coord, deg):
        x, y = coord
        rad = math.radians(deg)
        return {
            'x': (  cos(rad) * x + sin(rad) * y - hypot * 0.5 * cos(rad) - hypot * 0.5 * sin(rad) + hypot * 0.5 - (hypot - cols) * 0.5) / float(cols) * 100.0,
            'y': (- sin(rad) * x + cos(rad) * y + hypot * 0.5 * sin(rad) - hypot * 0.5 * cos(rad) + hypot * 0.5 - (hypot - rows) * 0.5) / float(rows) * 100.0,
        }
    # rotate and detect faces
    results = []
    for deg in range(-48, 49, 6):
        M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), deg, 1.0)
        rotated = cv2.warpAffine(frame, M, (hypot, hypot))
        faces = cascade_f.detectMultiScale(rotated, 1.08, 2)
        #faces = cascade_f.detectMultiScale(rotated, scaleFactor=1.2, minNeighbors=2, minSize=(50, 50))
        #print deg, len(faces)
        for face in faces:
            x, y, w, h = face
            #face_im = im[y:y+h, x:x+w]
            #cv2.imwrite('face/_face.jpg', face_im)
            # eyes in face?
            y_offset = int(h * 0.1)
            roi = rotated[y + y_offset: y + h, x: x + w]
            eyes = cascade_e.detectMultiScale(roi, 1.05)
            #eyes = filter(lambda e: (e[0] > w / 2 or e[0] + e[2] < w / 2) and e[1] + e[3] < h / 2, eyes)
            eyes = [e for e in eyes if (e[0] > w / 2 or e[0] + e[2] < w / 2) and e[1] + e[3] < h / 2]
            if len(eyes) == 2 and abs(eyes[0][0] - eyes[1][0]) > w / 4:
                score = math.atan2(abs(eyes[1][1] - eyes[0][1]), abs(eyes[1][0] - eyes[0][0]))
                if eyes[0][1] == eyes[1][1]:
                    score = 0.0
                results.append({
                    'center': translate([x + w * 0.5, y + h * 0.5], -deg),
                    'w': float(w) / float(cols) * 100.0,
                    'h': float(h) / float(rows) * 100.0,
                    'eyes': [translate([x + e[0] + e[2] * 0.5, y + y_offset + e[1] + e[3] * 0.5], -deg) for e in eyes],
                    'score': score,
                    'deg': deg+score,
                })
    # unify duplicate faces
    faces = []
    for result in results:
        x, y = result['center']['x'], result['center']['y']
        exists = False
        for i in range(len(faces)):
            face = faces[i]
            if (face['center']['x'] - face['w'] * 0.5 < x < face['center']['x'] + face['w'] * 0.5 and
                face['center']['y'] - face['h'] * 0.5 < y < face['center']['y'] + face['h'] * 0.5):
                exists = True
                if result['score'] < face['score']:
                    faces[i] = result
                    break
        if not exists:
            faces.append(result)
    for face in faces:
        del face['score']
    return faces

if __name__ == '__main__':
    if len(sys.argv) == 2:
        lists = glob.glob(sys.argv[1] + '/*')
        lists += glob.glob(sys.argv[1] + '/**/*')
        for index,f in enumerate(lists):
            print(f)
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            #im = mirror_padding(img)
            faces = detect(im)
            #print(faces)
            if len(faces) == 0:
                print(' face can not found!')
                continue

            rows, cols, _ = im.shape
            x = int(faces[0]['center']['x'] *cols/100 -(faces[0]['w'] *cols/100/2))
            y = int(faces[0]['center']['y'] *rows/100 -(faces[0]['h'] *rows/100/2))
            w = int(faces[0]['w'] *cols/100)
            h = int(faces[0]['h'] *rows/100)
            if x<0 or y<0:
                print(' strange ', x,y,w,h,faces[0])
                continue
            face_im = im[y:y+h, x:x+w]
            if face_im is None:
                print(' face can not found!')
                continue
            cv2.imwrite('face/'+ str(index) +'_face.jpg', face_im)
            #roi = cv2.resize(im[y: y + h, x: x + w], (64, 64), interpolation=cv2.INTER_LINEAR)

            #x_diff = int(faces[0]['eyes'][0]['x'] *cols/100 - faces[0]['eyes'][1]['x'] *cols/100)
            #y_diff = int(faces[0]['eyes'][0]['y'] *rows/100 - faces[0]['eyes'][1]['y'] *rows/100)
            #angle = math.degrees(math.atan2(y_diff, x_diff))
            #print(x_diff, y_diff, angle)
            angle = faces[0]['deg']
            rotated_im = fitting_rotated_image(face_im, angle)
            cv2.imwrite('face/rotated_face_'+ str(index) +'.jpg', rotated_im)

            #eyes = cascade_e.detectMultiScale(im[y: y + h, x: x + w])
            #if len(eyes) > 1:
                #face_im = im[y: y + h, x: x + w]
                #x_diff = int((eyes[0][0] + eyes[0][2] * 0.5) - (eyes[1][0] + eyes[1][2] * 0.5))
                #y_diff = int((eyes[0][1] + eyes[0][3] * 0.5) - (eyes[1][1] + eyes[1][3] * 0.5))
                #angle = math.degrees(math.atan2(y_diff, x_diff))
                #rotated_im = fitting_rotated_image(face_im, angle)
                #cv2.imwrite('face/rotated_face_'+ str(index) +'.jpg', rotated_im)

            #print(faces)
            #print(faces[0])
            #print(faces[0]['center']['x'])
            #print(faces[0]['center']['y'])
            #print(faces[0]['w'])
            #print(faces[0]['h'])
            #print(faces[0]['eyes'][0]['x'])
            #print(faces[0]['eyes'][0]['y'])
            #print(faces[0]['eyes'][1]['x'])
            #print(faces[0]['eyes'][1]['y'])
            #for (face) in faces[0].keys():
                #print(face)
            #for (face) in faces[0].values():
                #print(face)
            #cv2.imwrite('face/rotated_face_'+ str(index) +'.jpg', rotated_im)
    else:
        print('arguments error')

