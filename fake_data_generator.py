import pandas as pd
from faker import Faker

fake = Faker()

records = []
for _ in range(1000):
    records.append({
        "name": fake.name(),
        "email": fake.email(),
        "city": fake.city(),
        "job": fake.job(),
        "bio": fake.text(max_nb_chars=200)
    })

df = pd.DataFrame(records)
df.to_csv("users.csv", index=False)
print("Generated users.csv")
