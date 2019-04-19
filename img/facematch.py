import face_recognition

image_of_patrice = face_recognition.load_image_file('./img/known/Patrice Bergeron.jpg')
patrice_face_encoding = face_recognition.face_encodings(image_of_patrice)[0]

unknown_image = face_recognition.load_image_file('./img/unknown/jake-debrusk.jpg')
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# compare faces
results = face_recognition.compare_faces([patrice_face_encoding], unknown_face_encoding)

if results[0]:
    print('This is Patrice Bergeron')
else:
    print('This is NOT Patrice Bergeron')
