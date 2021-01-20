import falcon
import json
from speed_up_t5_onnx import OnnxT5
import wandb
import glob
import os
from transformers import T5Tokenizer

def get_model_from_w_b_without_run(artifact_name='T5_model_tasks:latest'):
    api = wandb.Api()
    artifact = api.artifact('blizd/mt-t5-ria-news/'+artifact_name, type='model')
    return artifact.download()

class ProcessMassge(object):
    def on_post(self, req, resp):
        message = req.media.get("message")
        resp.body = json.dumps({"test": "answer"})
        resp.status = falcon.HTTP_200
        return resp

path = get_model_from_w_b_without_run()
model_path = glob.glob(os.path.join(path, '*.pt'))[0]
tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
onnx_model = OnnxT5(model_name_or_path="google/mt5-small", onnx_path="onnx_models", model_st_d=model_path)
app = falcon.API()
message_processor = ProcessMassge()
app.add_route('/model', message_processor)