from utils import crop_image, compute_lines_pixels,find_cells
import cv2 as cv
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR



if __name__=='__main__':
    img=cv.imread(cv.samples.findFile('source img/2026-10-02_002.png'))

    cropped_img, cropped_org = crop_image(img)
    cv.imwrite('cropped_img.png',cropped_org)

    h_lines,h_grig=compute_lines_pixels(cropped_img,'h')
    v_lines,v_grig=compute_lines_pixels(cropped_img,'v')

    h_grig=cv.dilate(h_grig, np.ones((3,3), np.uint8))
    v_grig=cv.dilate(v_grig, np.ones((3,3), np.uint8))

    grid=cv.add(h_grig,v_grig)

    cropped_org[grid>0]=(255,255,255)

    # cv.imshow('crop',cropped_org)

    table=find_cells(h_lines,v_lines)

    # padding=10
    thresh=300

    cells=[]
    coordinates=[]
    for i,row in enumerate(table):
        table_row=[]
        for ii,cell in enumerate(row):
            img=cropped_org[cell[1]:cell[3],cell[0]:cell[2]]
            # img = cv.resize(img, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)
            # img=img[padding:img.shape[0],padding:img.shape[1]]
            # cv.imshow(f'row {i+1} cell {ii+1}',img)
            k=cv.waitKey(0)
            if k == ord('d'):
                pass
            cv.destroyAllWindows()

            gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            _,tresh_img=cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
            if cv.countNonZero(tresh_img)>thresh:
                # cv.imshow(f'row {i+1} cell {ii+1}',tresh_img)
                coordinates.append([i,ii])
            table_row.append(img)
        cells.append(table_row)


    table=pd.DataFrame(columns=['марка ТС','лом черных металллов, кг','лом цветных металлов, кг', 'ДВС',
       'Трам-блёр', 'Ген.', 'Стар-тер', 'БТ', 'КПП', 'РК', 'ПМ', 'СМ', 'ЗМ',
       'ПР', 'ЗР', 'Рама', 'РМ', 'Каби-на','Примечание'])

    ocr = PaddleOCR(lang='en',text_det_thresh=0.25)
    for i,ii in coordinates:
        img=cells[i][ii]
        # cv.imshow('cell',img)

        k=cv.waitKey(0)
        if k == ord('d'):
            pass
        cv.destroyAllWindows()
        if ii in [3,4,5,6,7,8,9,15,16]:
            table.loc[i,table.columns[ii]]=1
        else:
            result = ocr.predict(img)
            result_text=result[0].json['res']['rec_texts']
            if result_text:
                table.loc[i,table.columns[ii]]=result_text[0]
                print(result_text)
            else:
                print('cant find text')
    print(table)


