from datasets import load_dataset
import collections
data = load_dataset("./datasets/physics-bench-solid-eval")["train"]
# data = load_dataset("./datasets/physics-bench-dynamics-eval-1900")
# data = load_dataset("./datasets/physics-bench-optics-eval-2700")
# data = load_dataset("./datasets/physics-bench-solid-eval-2700")
# data = load_dataset("./datasets/physics-bench-optics-full-scale-2700")
# print(data["train"])
# print(data["train"][0],"\n\n")
# print(data["train"][1],"\n\n")
# print(data["train"][2],"\n\n")
# print(data["train"][3],"\n\n")
# print(data["train"][4],"\n\n")
# print(data["train"][50],"\n\n")
case_ids = data["case_id"]
case_ids = [case_id.split("_")[0] for case_id in case_ids]    
c = collections.Counter(case_ids)
print(c)

# for x in data["case_id"][:100]:
#     print(x)
# for x in range(700):
#     if "buoyancy" in data[x]["case_id"]:
#         print(data[x])
#     break


# print(data["train"][10],"\n\n")
# print(data["train"][11],"\n\n")
# print(data["train"][12],"\n\n")
# print(set(data["train"]["family"]))
# videos = data["train"]["video"]
# for video in videos[:10]:
#     print(video)