from pymongo.mongo_client import MongoClient 

uri = "mongodb+srv://nivakaran:LqXp6EI7OaM59r2Y@cluster0.aqkct2u.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a pin gto confirm a successful connnection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You are successfully connected to MongoDB!")
except Exception as e:
    print(e)

