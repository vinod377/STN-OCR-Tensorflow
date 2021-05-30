"""
The script takes digitStruct.mat file of SVHN, Reads it and save the labels and bbox
information python pickle file.The image is re-wriiten after cropping the image region
from full image and appending the label in the original image of the file.
"""
import cv2
import os
import h5py
import pickle

folder_path = "/dataset/svhn/train/"
write_folder = "/dataset/svhn/train_cropped/"

class Svhn:
    def __init__(self):
        pass
    def readingMatFile(self,file_path):
        attributes = ['height','label','left','top','width']
        f = h5py.File(file_path)
        num_images=f['/digitStruct/bbox'].shape
        annotation_dict = {}
        for key in attributes:
            key_list=[]
            img_cnt = 0
            for i in range(num_images[0]):
                ref= f['/digitStruct/bbox'][i,0]
                key_shape = f[ref][key].shape
                temp_list=[]
                img_cnt+=1

                temp_list.append(str(img_cnt) + '.png')
                for j in range(key_shape[0]):
                    height_ref = f[ref][key][j][0]
                    if key_shape[0]==1:
                        temp_list.append(height_ref)
                    else:
                        temp_list.append(f[height_ref][0,0])
                key_list.append(temp_list)
            annotation_dict[key]=key_list
        f=open('svhn_annotation_train_wb.pkl','wb')
        pickle.dump(annotation_dict,f)

    def displayAndWrite(self):
        svhn_file = open('svhn_annotation_train_wb.pkl','rb')
        data = pickle.load(svhn_file)
        labels = data['label'] #image label
        heights = data['height']
        widths = data['width']
        tops = data['top'] #y-cordinate
        lefts = data['left'] #x-coordinate
        for ind1 in range(len(labels)):
            file_name=labels[ind1][0]
            temp_label=[]
            coords = []
            for ind2 in range(1,len(labels[ind1])):
                if labels[ind1][ind2]==10:
                    val = 0
                else:
                    val=labels[ind1][ind2]
                temp_label.append(val)
                coords.append([int(lefts[ind1][ind2]),int(tops[ind1][ind2]),
                              int(lefts[ind1][ind2]+widths[ind1][ind2]),
                              int(tops[ind1][ind2]+heights[ind1][ind2])])
            print(file_name,temp_label,coords)
            file_path = os.path.join(folder_path,file_name)
            temp_label=[str(int(x)) for x in temp_label]
            temp_label=''.join(temp_label)

            image = cv2.imread(file_path)
            """
            when the digits in image are not at same height, we will be taking the lesser 
            Y value to cover the whole digit region
            """
            if coords[0][1]>coords[-1][3]:
                lower_y = coords[-1][3]
                higher_y = coords[0][1]
            else:
                lower_y = coords[0][1]
                higher_y = coords[-1][3]
            image_crp = image[lower_y:higher_y, max(coords[0][0],0):max(coords[-1][2],0)]
            print(image_crp.shape[0],image_crp.shape[1])
            image_name = os.path.join(write_folder, file_name[:-4] + '_x_' + temp_label + '.png')
            if image_crp.shape[0]==0 or image_crp.shape[1]==0:
                cv2.imwrite(image_name, image)
            else:
                cv2.imwrite(image_name, image_crp)
            # cv2.putText(image,temp_label,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),1)
            # cv2.imshow('image',image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

if __name__=="__main__":
    obj = Svhn()
    file_path = '/dataset/svhn/train/digitStruct.mat'
    # obj.readingMatFile(file_path)
    obj.displayAndWrite()


