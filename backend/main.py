import torch
from transformers import XLNetModel, XLNetTokenizer,AdamW
import torch.nn as nn



class XLNet_classifier(torch.nn.Module):

  def __init__(self, num_labels=2):
    super(XLNet_classifier, self).__init__()
    self.num_labels = num_labels
    self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
    self.classifier = torch.nn.Linear(768, num_labels)

    torch.nn.init.xavier_normal_(self.classifier.weight)

  def forward(self, input_ids, token_type_ids=None,\
              attention_mask=None, labels=None):
    # last hidden layer
    last_hidden_state = self.xlnet(input_ids=input_ids,\
                                   attention_mask=attention_mask,\
                                   token_type_ids=token_type_ids)
    # pool the outputs into a mean vector
    mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
    logits = self.classifier(mean_last_hidden_state)

    if labels is not None:
      loss_fct = BCEWithLogitsLoss()
      loss = loss_fct(logits.view(-1, self.num_labels),\
                      labels.view(-1, self.num_labels))
      return loss
    else:
      return logits


  def pool_hidden_state(self, last_hidden_state):

    last_hidden_state = last_hidden_state[0]
    mean_last_hidden_state = torch.mean(last_hidden_state, 1)
    return mean_last_hidden_state


model = XLNet_classifier(num_labels=6)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, correct_bias=False)
num_epochs=2


from transformers import XLNetTokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)


label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
model_save_path= "./xlnet_toxic.bin"
model = XLNet_classifier(num_labels=len(label_cols))
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu'))['state_dict'])

model.eval()

def pred(user_input):
    
    tokenized_input = tokenizer.encode(user_input, add_special_tokens=True)
    input_ids = torch.tensor(tokenized_input).unsqueeze(0)  # Add batch dimension
    attention_mask = torch.tensor([1] * len(tokenized_input)).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        logits = logits.sigmoid().detach().cpu().numpy()


    threshold = 0.5
    predicted_labels = (logits > threshold).astype(int)[0]

    toxic_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    output_labels = [label for label, pred in zip(toxic_labels, predicted_labels) if pred == 1]

    if len(output_labels) == 0:
        return "Non-toxic"
    else:
        return "Toxic"

from fastapi import FastAPI, HTTPException, Form

app = FastAPI()

from model import Comment

from database import (
    fetch_comments,
    fetch_all_comments,
    create_comment,
)


from fastapi.middleware.cors import CORSMiddleware


origins = [
    "http://localhost:3000",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}



@app.post("/api/comment/")
async def post_comment(comment: str = Form(...)):
    try:
        print(comment)
        label=pred(comment)

        new_doc = {
            "comment":comment,
            "label":label
        }
        response = await create_comment(new_doc)
        return response
    except YourValidationError as e:
        return {"error": str(e)}
 

@app.get("/api/comment/")
async def get_comment():
    response = await fetch_all_comments()
    return response

@app.get("/api/comment/{label}")
async def get_comment_by_label(label):
    print(label)
    response = await fetch_comments(label)
    if response:
        return response
    raise HTTPException(404, f"There is no {label} comments")
