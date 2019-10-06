# uvicorn --reload emotion-classifier:app

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse

from fastai.vision import (open_image, load_learner, learner)

from io import BytesIO
# import sys
import uvicorn
import aiohttp
# import asyncio


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette(debug=True)


# ----------------------------  LOAD LEARNER
file_path = 'emotion-classifier-model.pkl'
learner = load_learner('models',file_path)
# learner = load_learner(path, file_path)
print('learer.data.classes = ', learner.data.classes)


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)
    # return json_image_bytes(bytes)

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    print('open_imgage = ', img)
    # out_class, out_index_tensor, probabilities = learner.predict(img)
    # max_index = int(out_index_tensor) 

    # make a dictionary of classes with probabilities
    classes = learner.data.classes
    _, _, probabilities = learner.predict(img)
    out_classification = dict(zip( classes, map(float, probabilities) ))
    # sort by value to have best prediction first in order
    out_classification = sorted(out_classification.items(), key=lambda kv: kv[1], reverse=True)
    print('classification completed: ', out_classification)
    return JSONResponse({
        "predictions": out_classification
    })

@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)