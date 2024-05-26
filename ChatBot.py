import warnings 
warnings.filterwarnings('ignore')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-mini-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
intial_question_number=0
num_question = int(input("How many question you are to asked; "))
while intial_question_number<num_question:
  question = input("Enter your question: ")
  messages = [

      {"role": "user", "content": "You are an helpfull assistant your task is to answer questions."},
      {"role": "assistant", "content": "Sure!"},
      {"role": "user", "content": "what is photosynthesis?"},
      {"role": "assistant", "content": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy, typically from sunlight, into chemical energy in the form of sugars. This chemical energy is then used to fuel the organism's activities."},
      {"role": "user", "content": question},
  ]

  pipe = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
  )

  generation_args = {
      "max_new_tokens": 1000,
      "return_full_text": False,
      "do_sample": False,
  }

  output = pipe(messages, **generation_args)
  intial_question_number+=1
  print(f"@@@@@@@@@@@@@@@@@@@@@@@@@ Generating Answers@@@@@@@@@@@@@@@@@@@@@")
  print(output[0]['generated_text'])