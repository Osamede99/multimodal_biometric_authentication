# <SETUP> 
# Import libraries
from pathlib import Path
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# <END SETUP>

# <LOAD DATA>
DATA_DIRECTORY = Path(r"C:\path\to\AudioMNIST-master\data")

# Check whether the dataset folder is found
if not DATA_DIRECTORY.is_dir():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIRECTORY}")

print("Dataset found:", DATA_DIRECTORY.exists())

# <END LOAD DATA>

# <SELECT SPEAKERS>
# Select the first 40 speakers
PERSON_IDS = []

for i in range(1,41):
    PERSON_IDS.append(f"{i:02d}")
    
# <END SELECT SPEAKERS>

# <FEATURE EXTRACTION>
# Extract MFCC coefficients from an audio recording
def extract_features(audio, sample_rate):
    # Extract 13 MFCC coefficients 
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    
    # Calculate the mean and standard deviation
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Combine them into one feature vector
    features = np.concatenate((mfcc_mean, mfcc_std))
    
    return features
# <END FEATURE EXTRACTION>


# <EXTRACT DATASET FEATURE>
# Create lists to store features and person IDs
X = []
y = []
audio_paths = []

#Loop through all 40 people
for person_id in PERSON_IDS:
    person_path = DATA_DIRECTORY / person_id
    audio_files = sorted(person_path.glob("*.wav"))
    
    # Extract features from each recording
    for audio_file in audio_files:
        audio, sample_rate = librosa.load(audio_file, sr=None)
        
        features = extract_features(audio, sample_rate)
        
        X.append(features)
        y.append(int(person_id) - 1)
        audio_paths.append(str(audio_file))
        
# Convert lists into numpy arrays
X = np.array(X)
y = np.array(y)
audio_paths = np.array(audio_paths)

# Display the dataset shapes
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)
print("Audio paths shape:", audio_paths.shape)

# <END EXTRACT DATASET FEATURE>

# <SPLIT DATA>
# Split the dataset into training and testing sets
X_train, X_temp, y_train, y_temp, paths_train, paths_temp = train_test_split(
    X, y, audio_paths,
    test_size = 0.4,
    random_state = 42,
    stratify = y
    )

# Split the dataset into validation and testing sets
X_val, X_test, y_val, y_test, paths_val, paths_test = train_test_split(
    X_temp, y_temp, paths_temp,
    test_size = 0.5,
    random_state = 42,
    stratify = y_temp
    )

# Display the number of samples in each set
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
print("Training paths:", paths_train.shape)
print("Validation paths:", paths_val.shape)
print("Testing paths:", paths_test.shape)

# <END SPLIT DATA>

# <NORMALIZE DATA>
# Create a scaler for the MFCC features
scaler = StandardScaler()

# Fit the scaler using only the training data
X_train = scaler.fit_transform(X_train)

# Apply the same scaling to validation and testing data
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Display the normalized dataset shapes
print("Normalized training:", X_train.shape)
print("Normalized validation:", X_val.shape)
print("Normalized testing:", X_test.shape)

# <END NORMALIZE DATA>

# <SAVE DATA>
# Save processed datasets
filename = "voice_features.npz"
np.savez(filename, X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test,
         scaler_mean=scaler.mean_,
         scaler_scale=scaler.scale_)

print(f"File {filename} saved")

# <END SAVE DATA>

# <GAUSSIAN NOISE>
# Set random seed
np.random.seed(42)

def add_gaussian_noise(audio, noise_level):
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise

# Add Gaussian noise based on the target SNR
def add_noise_snr(audio, snr_db):
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power
    noise_power = signal_power / (10 ** (snr_db/10))

    # Calculate noise standard deviation
    noise_std = np.sqrt(noise_power)

    # Add Gaussian noise 
    noisy_audio = add_gaussian_noise(audio, noise_std)
    
    return noisy_audio

# <END GAUSSIAN NOISE>

# <GENERATE NOISY FEATURES>
def generate_noisy_features(audio_paths, snr_db):
    X_noisy = []
    
    # Load each audio recording
    for audio_path in audio_paths:
        audio, sample_rate = librosa.load(audio_path, sr=None)
        
        #Add Gaussian noise at the specified SNR
        noisy_audio = add_noise_snr(audio, snr_db)
        
        # Extract MFCC features
        features = extract_features(noisy_audio, sample_rate)
        
        X_noisy.append(features)
        
    X_noisy = np.array(X_noisy)
    
    X_noisy = scaler.transform(X_noisy)
    
    return X_noisy

# <END GENERATE NOISY FEATURES>
    

# <20DB NOISY VALIDATION DATA>
# Generate noisy validation features at 20 dB
X_val_noisy_20dB = generate_noisy_features(paths_val, 20)

print("20 dB noisy validation:", X_val_noisy_20dB.shape)
# <END 20DB NOISY VALIDATION DATA>

# <10DB NOISY VALIDATION DATA>
# Generate noisy validation features at 10 dB
X_val_noisy_10dB = generate_noisy_features(paths_val, 10)

print("10 dB noisy validation:", X_val_noisy_10dB.shape)

# <END 10DB NOISY VALIDATION DATA>

# <20DB NOISY TESTING DATA>
# Generate noisy testing features at 20dB
X_test_noisy_20dB = generate_noisy_features(paths_test, 20)

print("20 dB noisy testing:", X_test_noisy_20dB.shape)
# <END 20DB NOISY TESTING DATA>

# <10DB NOISY TESTING DATA>
# Generate noisy testing features at 10dB
X_test_noisy_10dB = generate_noisy_features(paths_test, 10)

print("10 dB noisy testing:", X_test_noisy_10dB.shape)
# <END 10DB NOISY TESTING DATA>

# <SAVE NOISY DATA>
filename = "voice_noisy_features.npz"

np.savez(
    filename,
    X_val_noisy_20dB=X_val_noisy_20dB,
    X_val_noisy_10dB=X_val_noisy_10dB,
    X_test_noisy_20dB=X_test_noisy_20dB,
    X_test_noisy_10dB=X_test_noisy_10dB,
    y_val=y_val,
    y_test=y_test)

print(f"File {filename} saved")

# <END SAVE NOISY DATA>

