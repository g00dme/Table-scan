from utils import crop_image, compute_lines_pixels,find_cells
import cv2 as cv
import pytesseract 


if __name__=='__main__':
    img=cv.imread(cv.samples.findFile('table.png'))

    cropped_img, cropped_org = crop_image(img)
    cv.imwrite('cropped_img.png',cropped_org)

    h_lines=compute_lines_pixels(cropped_img,'h')
    v_lines=compute_lines_pixels(cropped_img,'v')
    cv.imshow('crop',cropped_img)

    table=find_cells(h_lines,v_lines)
    cells=[]
    for i,row in enumerate(table):
        table_row=[]
        for ii,cell in enumerate(row):
            img=cropped_org[cell[1]:cell[3],cell[0]:cell[2]]
            cv.imshow(f'row {i+1} cell {ii+1}',img)
            table_row.append(img)
        cells.append(table_row)
    img=cells[0][0]

    img = cv.resize(img, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)

    # cv.imshow('cell',img)


    k=cv.waitKey(0)
    if k == ord('r'):
        print('ol3g')
    cv.destroyAllWindows()

    # text= pytesseract.image_to_string(img, lang='rus')
    # print(text)
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='ru')
    result = ocr.predict(img)
    print(result)
