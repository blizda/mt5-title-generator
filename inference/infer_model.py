import falcon
import json
from speed_up_t5_onnx import OnnxT5

class ProcessMassge(object):
    def on_post(self, req, resp):
        message = req.media.get("message")
        resp.body = json.dumps({"test": "answer"})
        resp.status = falcon.HTTP_200
        return resp

app = falcon.API()
message_processor = ProcessMassge()
app.add_route('/model', message_processor)