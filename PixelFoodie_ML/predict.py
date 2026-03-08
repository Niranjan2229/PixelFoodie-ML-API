import pickle

with open('pixel_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Example: Spicy(5), Time(40), Non-Veg(0), Snacks(3), SouthIndian(1)
user_input = [[1, 40, 0, 3, 1]] 
result = loaded_model.predict(user_input)

print(f"🌟 PixelFoodie AI Suggestion: {result[0]}")