import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
import pickle


def capture_images(name, num_images):
    cap = cv2.VideoCapture(0)
    os.makedirs(name, exist_ok=True)
    print(f"Capturing images for {name}...")
    for i in range(num_images):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue
        cv2.imshow("frame", frame)
        cv2.imwrite(f"{name}/{i}.jpg", frame)
        time.sleep(0.5)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def train_model():
    data = []
    labels = []
    for person in ["person1", "person2"]:
        for filename in os.listdir(person):
            img = cv2.imread(os.path.join(person, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                data.append(img.flatten())
                labels.append(person)
    data = np.array(data)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    print("Training the model...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print("Model trained successfully!")
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    person1_name = input("Enter the name for person 1: ")
    person2_name = input("Enter the name for person 2: ")
    model_filename = f"{person1_name}_vs_{person2_name}_model.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(knn, file)
    print(f"Model saved as {model_filename}")

    return knn


def identify_person(model):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = gray.flatten().reshape(1, -1)
    person = model.predict(face)[0]
    print(f"Identified person: {person}")
    cap.release()


def get_model_files():
    model_files = [file for file in os.listdir() if file.endswith("_model.pkl")]
    return model_files


def main():
    while True:
        print("\nFacial Recognition Menu:")
        print("1. Capture images for person 1")
        print("2. Capture images for person 2")
        print("3. Train the model")
        print("4. Identify person")
        print("5. Quit")
        choice = input("Enter your choice (1-5): ")
        if choice == "1":
            capture_images("person1", 100)
        elif choice == "2":
            capture_images("person2", 100)
        elif choice == "3":
            model = train_model()
        elif choice == "4":
            model_files = get_model_files()
            if len(model_files) == 0:
                print("No trained models found. Please train the model first.")
            else:
                print("Available models:")
                for i, file in enumerate(model_files, start=1):
                    print(f"{i}. {file}")
                model_choice = int(
                    input("Enter the number of the model you want to use: ")
                )
                if 1 <= model_choice <= len(model_files):
                    model_filename = model_files[model_choice - 1]
                    with open(model_filename, "rb") as file:
                        model = pickle.load(file)
                    identify_person(model)
                else:
                    print("Invalid model choice.")
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
