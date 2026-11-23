# =============================================
# 🐾 PET MATCHER 2.0
# Find Your Perfect Animal Companion!
# =============================================

# STEP 1: Import libraries

import pandas as pd
import ipywidgets as widgets

from IPython.display import display, clear_output
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier



# =============================================
# 🐾 PET MATCHER 2.0
# Find Your Perfect Animal Companion!
# =============================================

# STEP 1: Import libraries

import pandas as pd
import ipywidgets as widgets

from IPython.display import display, clear_output
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


# STEP 2: Create pet profiles

# Each pet has a typical lifestyle profile.
# Values range from 1 to 3.

pet_profiles = {
    "Dog":         [3, 3, 3, 3, 3, 3, "mammal"],
    "Cat":         [2, 2, 2, 2, 1, 2, "mammal"],
    "Rabbit":      [2, 2, 2, 2, 1, 3, "mammal"],
    "Hamster":     [1, 1, 1, 1, 1, 2, "mammal"],
    "Guinea pig":  [2, 2, 2, 3, 2, 3, "mammal"],
    "Bird":        [2, 2, 3, 3, 3, 3, "bird"],
    "Fish":        [1, 1, 2, 1, 1, 2, "fish"],
    "Turtle":      [1, 2, 2, 1, 1, 2, "reptile"],
    "Leopard gecko":[1, 1, 1, 1, 1, 2, "reptile"],
    "Ferret":      [3, 2, 3, 3, 2, 3, "mammal"]
}


# STEP 3: Generate example training data

columns = [
    "activity",
    "space",
    "time",
    "interaction",
    "noise",
    "routine",
    "animal_group"
]

rows = []

# Create four slightly different examples for every pet.
variations = [
    (0, 0, 0),
    (0, 0, 1),
    (1, 0, 0),
    (0, 1, 0)
]

for pet, profile in pet_profiles.items():

    for variation in variations:

        example = profile.copy()

        # Add small variations to activity, space and time
        for i, change in enumerate(variation):
            example[i] = max(1, min(3, example[i] + change))

        rows.append(example + [pet])


df = pd.DataFrame(rows, columns=columns + ["pet"])

print("🐾 Pet Matcher Dataset")
display(df.head(10))

print("Total training examples:", len(df))


# STEP 4: Separate input features and target

X = df[columns]
y = df["pet"]


# STEP 5: Preprocess the data

numeric_features = [
    "activity",
    "space",
    "time",
    "interaction",
    "noise",
    "routine"
]

categorical_features = ["animal_group"]

preprocessor = ColumnTransformer([
    ("numbers", StandardScaler(), numeric_features),

    ("categories",
     OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])


# STEP 6: Build the KNN model

model = Pipeline([
    ("preprocessing", preprocessor),

    ("knn", KNeighborsClassifier(
        n_neighbors=3,
        weights="distance"
    ))
])


# STEP 7: Train the model

model.fit(X, y)

print("\n✅ Machine learning model trained successfully!")


# STEP 8: Fun pet descriptions

pet_descriptions = {

    "Dog": {
        "description":
        "Your perfect companion might be a dog! 🐶 "
        "Loyal, playful, and full of energy, dogs can make "
        "every ordinary day feel like an adventure.",

        "care":
        "Dogs need daily exercise, training, social interaction, "
        "veterinary care, and consistent attention."
    },

    "Cat": {
        "description":
        "Meet your potential feline best friend! 🐱 "
        "Cats are curious, independent, and masters of cozy "
        "afternoons. They can be affectionate while enjoying "
        "their personal space.",

        "care":
        "Cats need daily play, enrichment, veterinary care, "
        "and a clean litter box."
    },

    "Rabbit": {
        "description":
        "A fluffy little friend might be waiting for you! 🐰 "
        "Rabbits are gentle, curious, and can develop strong "
        "bonds with their humans.",

        "care":
        "Rabbits need spacious housing, daily exercise, "
        "unlimited hay, enrichment, and specialized veterinary care."
    },

    "Hamster": {
        "description":
        "Small but full of personality! 🐹 "
        "Hamsters love exploring, digging, and creating "
        "their own tiny adventures.",

        "care":
        "Hamsters need a spacious enclosure, deep bedding, "
        "an appropriate exercise wheel, and enrichment. "
        "They are usually active at night."
    },

    "Guinea pig": {
        "description":
        "Congratulations! You might have found your little "
        "chatty companion! 🐹 Guinea pigs are social, gentle, "
        "and famous for their adorable squeaks.",

        "care":
        "Guinea pigs generally need another compatible guinea pig, "
        "spacious housing, vitamin C, hay, and daily care."
    },

    "Bird": {
        "description":
        "A feathered friend could brighten your world! 🐦 "
        "Birds are intelligent, expressive, and often very "
        "social. Some can even learn sounds and tricks!",

        "care":
        "Birds need social interaction, a spacious safe environment, "
        "mental stimulation, and species-specific care."
    },

    "Fish": {
        "description":
        "Your ideal companion might live underwater! 🐟 "
        "Fish can bring a peaceful atmosphere to your home "
        "and are fascinating to observe.",

        "care":
        "Fish need a properly sized and cycled aquarium, "
        "appropriate filtration, water-quality monitoring, "
        "and regular maintenance."
    },

    "Turtle": {
        "description":
        "Slow and steady might be your perfect match! 🐢 "
        "Turtles are fascinating animals with unique behaviors "
        "and surprisingly specific needs.",

        "care":
        "Turtles need species-appropriate habitats, specialized "
        "lighting, temperature control, and long-term veterinary care."
    },

    "Leopard gecko": {
        "description":
        "Meet the tiny dinosaur of your dreams! 🦎 "
        "Leopard geckos are fascinating reptiles with "
        "distinctive patterns and curious personalities.",

        "care":
        "They need a properly heated enclosure, hiding spaces, "
        "appropriate supplements, and live insect feeding."
    },

    "Ferret": {
        "description":
        "Get ready for a mischievous little explorer! "
        "Ferrets are playful, curious, and love investigating "
        "every corner of their surroundings. 🦦",

        "care":
        "Ferrets need substantial daily interaction, "
        "a large secure habitat, supervised play, and "
        "specialized veterinary care."
    }
}


# STEP 9: Create interactive questions

print("\n🐾 WELCOME TO PET MATCHER 2.0!")
print("Let's discover your ideal animal companion!\n")


activity_widget = widgets.Dropdown(
    options=[
        ("Low - I enjoy a relaxed lifestyle", 1),
        ("Medium - I stay moderately active", 2),
        ("High - I love being active", 3)
    ],
    description="Activity:",
    style={"description_width": "initial"}
)


space_widget = widgets.Dropdown(
    options=[
        ("Small - Apartment or small room", 1),
        ("Medium - Average living space", 2),
        ("Large - Spacious home", 3)
    ],
    description="Living space:",
    style={"description_width": "initial"}
)


time_widget = widgets.Dropdown(
    options=[
        ("1 - Less than 1 hour daily", 1),
        ("2 - Around 1–3 hours daily", 2),
        ("3 - More than 3 hours daily", 3)
    ],
    description="Available time:",
    style={"description_width": "initial"}
)


interaction_widget = widgets.Dropdown(
    options=[
        ("Independent - I enjoy my own space", 1),
        ("Balanced - A little of both", 2),
        ("Social - I love interaction", 3)
    ],
    description="Interaction:",
    style={"description_width": "initial"}
)


noise_widget = widgets.Dropdown(
    options=[
        ("Quiet - I prefer peaceful surroundings", 1),
        ("Moderate - Some noise is okay", 2),
        ("High - I don't mind vocal pets", 3)
    ],
    description="Noise tolerance:",
    style={"description_width": "initial"}
)


routine_widget = widgets.Dropdown(
    options=[
        ("Unpredictable - My schedule changes", 1),
        ("Somewhat predictable", 2),
        ("Very consistent - I have a routine", 3)
    ],
    description="Daily routine:",
    style={"description_width": "initial"}
)


preference_widget = widgets.Dropdown(
    options=[
        ("Any animal - Surprise me!", "any"),
        ("Mammals", "mammal"),
        ("Birds", "bird"),
        ("Fish", "fish"),
        ("Reptiles", "reptile")
    ],
    description="Animal group:",
    style={"description_width": "initial"}
)


# STEP 10: Recommendation function

def find_pet(button):

    with output:

        clear_output()

        # Collect answers into a DataFrame

        user_answers = pd.DataFrame([{
            "activity": activity_widget.value,
            "space": space_widget.value,
            "time": time_widget.value,
            "interaction": interaction_widget.value,
            "noise": noise_widget.value,
            "routine": routine_widget.value,
            "animal_group": preference_widget.value
        }])

        # Predict the best matching pet

        prediction = model.predict(user_answers)[0]

        # Display the recommendation

        print("=" * 45)
        print("       🐾 YOUR PET MATCH IS HERE! 🐾")
        print("=" * 45)

        print("\n✨ YOUR MATCH:", prediction.upper())

        print("\n" + pet_descriptions[prediction]["description"])

        print("\n💚 WHY THIS MATCH?")
        print(
            "Our KNN model compared your seven lifestyle "
            "features with similar pet profiles in its dataset."
        )

        print("\n📋 YOUR PET CARE GUIDE")
        print(pet_descriptions[prediction]["care"])

        print("\n" + "=" * 45)
        print("Remember: A match is only a starting point!")
        print("Research the animal's needs before adopting.")
        print("=" * 45)


# STEP 11: Create recommendation button

button = widgets.Button(
    description="Find My Perfect Pet! 🐾",
    button_style="success",
    layout=widgets.Layout(width="250px", height="45px")
)

output = widgets.Output()

button.on_click(find_pet)


# STEP 12: Display the application

print("\nAnswer all seven questions below:\n")

display(activity_widget)
display(space_widget)
display(time_widget)
display(interaction_widget)
display(noise_widget)
display(routine_widget)
display(preference_widget)

display(button)
display(output)