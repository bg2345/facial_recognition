import face_recognition
from PIL import Image, ImageDraw

image_of_patrice = face_recognition.load_image_file('./img/known/Patrice Bergeron.jpg')
patrice_face_encoding = face_recognition.face_encodings(image_of_patrice)[0]

image_of_brad = face_recognition.load_image_file('./img/known/Brad Marchand.jpg')
brad_face_encoding = face_recognition.face_encodings(image_of_brad)[0]

# create array of encodings and names
known_face_encodings = [
    patrice_face_encoding,
    brad_face_encoding
]

known_face_names = [
    "Patrice Bergeron",
    "Brad Marchand"
]

# load test image
test_image = face_recognition.load_image_file('./img/groups/patrice-brad.jpeg')

# find faces
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# convert to PIL format
pil_image = Image.fromarray(test_image)

# create an ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown Person"

    # if match
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(255,0,0))

    # draw label
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left - 12, bottom - text_height + 50), (right + 50, bottom + 50)), fill=(0,0,0), outline=(255,0,0))
    draw.text((left - 6, bottom - text_height + 50), name, fill=(255,255,255,255))

del draw

# display image
pil_image.show()

# save image
pil_image.save('identify.jpg')
