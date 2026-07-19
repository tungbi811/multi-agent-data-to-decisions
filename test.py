from agents.group_chat import GroupChat

group_chat = GroupChat()
events = group_chat.run(
    dataset_paths="data/bank_churn/train.csv",
    user_requirements="""
        Please help me to analyse this dataset.
    """
)

for event in events:
    print(event)