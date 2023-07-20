# import required libraries
import ssl
import cv2
from waitress import serve
# from uvicorn import Config, SSLConfig
import numpy as np
from io import BytesIO
import face_recognition
# import oracledb
import cx_Oracle
import pandas as pd
from PIL import Image
import base64
import tkinter as tk
from PIL import Image as im
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile,Form
from rembg import remove
import io
# import uvicorn.workers.UvicornWorker
from fastapi import FastAPI, Request,Body
import uvicorn
import json
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
# cx_Oracle.init_oracle_client(lib_dir="C:\oracle_installation\instantclient_21_9")
cx_Oracle.init_oracle_client(lib_dir="C:\Oracle instant\instantclient_21_9")
# class Face(BaseModel):
#     Employee_ID: str
# from flask import Flask, request
app=FastAPI()
# dsn = cx_Oracle.makedsn(
#         'localhost',
#         '1521',
#         service_name='orcl.mpcz.in'
# )
# conn = cx_Oracle.connect(
#         user='SYSTEM',
#         password='SYSTEM',
#         dsn=dsn
# )
dsn = cx_Oracle.makedsn(
    'localhost',
    '1521',
    service_name='orcl'
)
conn = cx_Oracle.connect(
    user='ambikesh_singh',
    password='MyPassword',
    dsn=dsn
)
# c = conn.cursor()
@app.post("/Attendence")
async def getResponse(Employee_ID: str,image: UploadFile = File(...)):

    c = conn.cursor()
    # print(Employee_ID)
    # Update_consumer = data['Employee_ID']
    # Read the contents of the uploaded image file
    c.execute('SELECT "EMPLOYEE_ID" from "FACE_DATA" where "EMPLOYEE_ID" =' + Employee_ID + '')
    e_id = c.fetchall()
    if(len(e_id)==1):

        contents = await image.read()
        # contents = await original_image.read()
        print(type(contents), "((((((((((((((((((((9")
        # Create a BytesIO object to hold the byte data
        byte_stream = io.BytesIO(contents)
        # Open the image from the BytesIO object \
        image = Image.open(byte_stream)
        # Display or save the image as desired
        # image.show()  # Display the image
        image.save('att_1.jpeg')  # Save the image to a file
        # Read the image file
        image = cv2.imread("att_1.jpeg")
        # Convert the image to binary data
        _, data = cv2.imencode(".png", image)
        # Pass the binary data to remove()
        out = remove(data.tobytes())
        # Convert the output binary data back to an image
        output_image = np.frombuffer(out, dtype=np.uint8)
        output_image = cv2.imdecode(output_image, cv2.IMREAD_UNCHANGED)
        cv2.imwrite("att_3.jpeg", output_image)
        # Open the image file
        image = Image.open("att_3.jpeg")

        # Create a BytesIO object
        byte_stream = io.BytesIO()

        # Save the image to the BytesIO object as bytes
        image.save(byte_stream, format='JPEG')

        # Get the byte data from the BytesIO object
        image_bytes = byte_stream.getvalue()


        # Insert the image and ID into the Oracle database
        query = "UPDATE FACE_DATA SET CURRENT_IMAGE = :blobdata WHERE EMPLOYEE_ID = :employee_id"
        c.execute(query, {"blobdata": image_bytes, "employee_id": Employee_ID})
        # c.execute("INSERT INTO FACE_RECOGNITION (EMPLOYEE_ID, CURRENT_IMAGE) VALUES (:1, :2)", (Employee_ID, contents))
        conn.commit()

        c.execute('SELECT "EMPLOYEE_ID" from "FACE_DATA" where "EMPLOYEE_ID" =' + Employee_ID + '')
        e_id=c.fetchall()
        i = 0
        while i < len(e_id):
            # print(e_id[i])
            new_data = e_id[i]
            i += 1
            c = conn.cursor()
            c.execute('SELECT "CURRENT_IMAGE" FROM "FACE_DATA" where "EMPLOYEE_ID" =' + Employee_ID + '')
            record = c.fetchall()
            # print(record)
            import base64
            # with open("ad_1.JPEG", "rb") as image2string:
            converted_string = base64.b64encode(record[0][0].read())
            # print(converted_string)

            with open('Current_image_encode.bin', "wb") as file:
                file.write(converted_string)


            file = open('Current_image_encode.bin', 'rb')
            byte = file.read()
            file.close()

            decodeit = open('Current_image.jpeg', 'wb')
            decodeit.write(base64.b64decode((byte)))
            decodeit.close()

            img = cv2.imread('Current_image.jpeg')
            # plt.imshow(img)
            # plt.show()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # read haar cascade for face detection
            # face_cascade = cv2.CascadeClassifier(r'C:\Users\ritesh.s\PycharmProjects\opencv9_project\opencv9_venv\haarcascade_frontalface_default.xml')
            #
            # # read haar cascade for smile detection
            # smile_cascade = cv2.CascadeClassifier(r'C:\Users\ritesh.s\PycharmProjects\opencv9_project\opencv9_venv\haarcascade_smile.xml')
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            # read haar cascade for smile detection
            smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

            # Detects faces in the input image
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            print('Number of detected faces:', len(faces))
            if(len(faces)==1):


            # loop over all the faces detected
                for (x, y, w, h) in faces:
            # print("66666666")
                # draw a rectangle in a face
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(img, "Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]

                    # detecting smile within the face roi
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.3, 10)
                if len(smiles) >0:

                train_elon_encodings = face_recognition.face_encodings(img)[0]
                c.execute('SELECT "ORIGINAL_IMAGE" FROM "FACE_DATA" where "EMPLOYEE_ID" =' + Employee_ID + '')
                result = c.fetchone()
                    # print(result, "&&&&&&&&&&&&&")
                image_data = result[0].read()

            # create a PIL Image object from the image data
                image = Image.open(BytesIO(image_data))

            # save the image to a file
                file_name = "original_image.jpeg"
                image.save(file_name)
                with open("original_image.jpeg", "rb") as f:
                    img_data = f.read()
                with open("Current_image.jpeg","rb") as f:
                    curr_img_data=f.read()

            # load the image from file and get its encoding
                test = face_recognition.load_image_file("original_image.jpeg")

                test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
                test_encode = face_recognition.face_encodings(test)[0]
                status=face_recognition.compare_faces([train_elon_encodings], test_encode, tolerance= 0.4)
                    # print(status,"%%%%%%%%%%%")
                    # print(type(status))
                if(status==[True]):
                    # print("testing 2")
                    # query = 'INSERT INTO "FACE_RECOGNITION" (EMPLOYEE_ID, CURRENT_IMAGE) VALUES (:EMPLOYEE_ID, :blobdata)'
                    query="""UPDATE FACE_DATA SET CURRENT_IMAGE = :blobdata WHERE EMPLOYEE_ID = :EMPLOYEE_ID"""
                    c.execute(query, {'blobdata': curr_img_data, 'EMPLOYEE_ID': Employee_ID})
                    # print(c.execute,"$$$$$$$$")


                    # Commit the transaction
                    conn.commit()
                    # conn.close()


                    # print("Attandance has been marked")
                    return {"status_code": 200,
                                "Attandance": "Attendance marked successfully",
                                "Message":"Face matched)",
                                "Face_Matched": "TRUE"}


                else:
                    return { "Message":"Face did not matched !",
                                "status_code": 201,
                                "Face_Matched": "FALSE",
                                }

            else:
                return {"Message": "No Face or Multiple Faces detected",
                            "status_code": 201,
                            "Face_Matched": "FALSE"
                            }

            break

    else:
        return {"Message": "Employee not registered",
                "status_code": 201}



# else:
            #             img = cv2.imread("Current_image.jpeg")
            #             with open("Current_image.jpeg","rb") as f:
            #                 curr_img_data=f.read()
            #         # plt.imshow(img)
            #         # plt.show()
            #             train_elon_encodings = face_recognition.face_encodings(img)[0]
            #         # print(train_elon_encodings)
            #         # print("testing else part")
            #             c.execute('SELECT "ORIGINAL_IMAGE" FROM "FACE_DATA" where "EMPLOYEE_ID" =' + Employee_ID + '')
            #             result = c.fetchone()
            #         # print(result, "&&&&&&&&&&&&&")
            #             image_data = result[0].read()
            #
            #         # create a PIL Image object from the image data
            #             image = Image.open(BytesIO(image_data))
            #
            #         # save the image to a file
            #             file_name = "e_original_image.jpeg"
            #             image.save(file_name)
            #             with open("e_original_image.jpeg", "rb") as f:
            #                 img_data = f.read()
            #
            #         # load the image from file and get its encoding
            #             test = face_recognition.load_image_file("e_original_image.jpeg")
            #             test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
            #             test_encode = face_recognition.face_encodings(test)[0]
            #             status = face_recognition.compare_faces([train_elon_encodings], test_encode, tolerance= 0.4)
            #         # print(status, "&&&&&&&&")
            #         # print(type(status))
            #             if (status == [True]):
            #             # print("testing 2")
            #         # query = 'INSERT INTO "FACE_RECOGNITION" (EMPLOYEE_ID, CURRENT_IMAGE) VALUES (:EMPLOYEE_ID, :blobdata)'
            #                 query = """UPDATE FACE_DATA SET CURRENT_IMAGE = :blobdata WHERE EMPLOYEE_ID = :EMPLOYEE_ID"""
            #                 c.execute(query, {'blobdata': curr_img_data, 'EMPLOYEE_ID': Employee_ID})
            #         # print(c.execute,"$$$$$$$$")
            #
            #         # Commit the transaction
            #                 conn.commit()
            #         # conn.close()
            #
            #         # print("Attandance has been marked")
            #                 return {"status_code": 200,
            #                     "Attandance":"Attendance marked successfully",
            #                     "Message": "Face matched. Smile not detected",
            #                     "Face_Matched": "TRUE"}
            #             else:
            #                 return {"Message": "Face did not matched !",
            #                 "status_code": 201,
            #                 "Face_Matched": "FALSE",
            #                 }
            #
            #
            #     # return {"smile": "Smile has not  been Detected",
            #     #         "status code": 200,
            #
            #     #         }
    #         else:
    #             return {"Message": "No Face or Multiple Faces detected",
    #                 "status_code": 201,
    #                 "Face_Matched": "FALSE"
    #                 }
    #
    #         break
    # else:
    #     return {"Message": "Employee not registered",
    #                 "status_code": 201}






@app.post("/Registration")
async def Registration(Employee_ID: str,original_image: UploadFile = File(...)):
    # data = jsonable_encoder(data)
    # dsn = cx_Oracle.makedsn(
    #     'localhost',
    #     '1521',
    #     service_name='orcl')
    # conn = cx_Oracle.connect(
    #     user='SYSTEM',
    #     password='SYSTEM',
    #     dsn=dsn)

    c = conn.cursor()
    contents = await original_image.read()
    print(type(contents),"((((((((((((((((((((9")
    # Create a BytesIO object to hold the byte data
    byte_stream = io.BytesIO(contents)
    # Open the image from the BytesIO object
    image = Image.open(byte_stream)
    # Display or save the image as desired
    # image.show()  # Display the image
    image.save('reg_1.jpeg')  # Save the image to a file
    # Read the image file
    image = cv2.imread("reg_1.jpeg")
    # Convert the image to binary data
    _, data = cv2.imencode(".png", image)
    # Pass the binary data to remove()
    out = remove(data.tobytes())
    # Convert the output binary data back to an image
    output_image = np.frombuffer(out, dtype=np.uint8)
    output_image = cv2.imdecode(output_image, cv2.IMREAD_UNCHANGED)
    cv2.imwrite("reg_2.jpeg", output_image)
    # Open the image file
    image = Image.open("reg_2.jpeg")

    # Create a BytesIO object
    byte_stream = io.BytesIO()

    # Save the image to the BytesIO object as bytes
    image.save(byte_stream, format='JPEG')

    # Get the byte data from the BytesIO object
    image_bytes = byte_stream.getvalue()


    c.execute('SELECT "EMPLOYEE_ID" from "FACE_DATA" where "EMPLOYEE_ID" =' + Employee_ID + '')
    e_id = c.fetchall()
    if(len(e_id)==0):



        # Insert the image and ID into the Oracle database
        c.execute("INSERT INTO FACE_DATA (EMPLOYEE_ID, ORIGINAL_IMAGE) VALUES (:1, :2)", (Employee_ID, image_bytes))
        conn.commit()
        # Update_consumer = data['Employee_ID']

        c.execute('SELECT "EMPLOYEE_ID" from "FACE_DATA" where "EMPLOYEE_ID" =' + Employee_ID + '')
        e_id=c.fetchall()
        i = 0
        while i < len(e_id):
            # print(e_id[i])
            new_data = e_id[i]
            i += 1
            c = conn.cursor()
            c.execute('SELECT "ORIGINAL_IMAGE" FROM "FACE_DATA" where "EMPLOYEE_ID" =' + Employee_ID + '')
            record = c.fetchall()
            # print(record)
            import base64
        # with open("ad_1.JPEG", "rb") as image2string:
            converted_string = base64.b64encode(record[0][0].read())

            with open('R_ORIGINAL_encode.bin', "wb") as file:        # print(converted_string)

                file.write(converted_string)

            # import base64

            file = open('R_ORIGINAL_encode.bin', 'rb')
            byte = file.read()
            file.close()

            decodeit = open('R_ORIGINAL_IMAGE.jpeg', 'wb')
            decodeit.write(base64.b64decode((byte)))
            decodeit.close()

            img = cv2.imread('R_ORIGINAL_IMAGE.jpeg')
            # plt.imshow(img)
            # plt.show()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # read haar cascade for face detection
            face_cascade = cv2.CascadeClassifier(r'C:\Users\ritesh.s\PycharmProjects\opencv9_project\opencv9_venv\haarcascade_frontalface_default.xml')

            # read haar cascade for smile detection
            smile_cascade = cv2.CascadeClassifier(r'C:\Users\ritesh.s\PycharmProjects\opencv9_project\opencv9_venv\haarcascade_smile.xml')
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            # read haar cascade for smile detection
            smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
            # Detects faces in the input image
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            print('Number of detected faces:', len(faces))
            if(len(faces)==1):

            # loop over all the faces detected
                for (x, y, w, h) in faces:
            # print("66666666")
                # draw a rectangle in a face
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(img, "Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]

                # detecting smile within the face roi
                    smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                    smiles = smile_cascade.detectMultiScale(roi_gray, 1.3, 10)
                    if (len(smiles) >0):
                        print("%%%%%%%%%%%%%%%")
                return{"status":200,
                        "message":"Employee Registered Successfully"}
            else:
                delete_query = "DELETE FROM FACE_DATA WHERE EMPLOYEE_ID = :emp_id"

                        # define the value to be used in the where condition
                where_value = {'emp_id': Employee_ID}

                        # execute the query with the where condition value
                c.execute(delete_query, where_value)

                        # commit the changes to the database
                conn.commit()

                return{"status": 201,
                    "message": "No Face or Multiple Faces detected"}
        else:
            delete_query = "DELETE FROM FACE_DATA WHERE EMPLOYEE_ID = :emp_id"

                # define the value to be used in the where condition
            where_value = {'emp_id': Employee_ID}

                # execute the query with the where condition value
            c.execute(delete_query, where_value)

                # commit the changes to the database
            conn.commit()

            return {"status": 201,
                    "message": "No Face or Multiple Faces detected"}
        break
    else:
        return {"status": 200,
                "message": "Employee Already registered"}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

