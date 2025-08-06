# HearEmotions  
*Helping Everyone Understand Sounds – Even Without Hearing Them*

## What’s the Goal?  
HearEmotions is a tool designed to help deaf and hard-of-hearing people understand what’s happening around them through sound. It doesn’t just detect a sound—it explains what the sound is and why it might be happening.  
For example:  
- “Your dog is barking. It looks like the dog is scared.”  
- “Your baby is crying. The baby is hungry.”  

This makes sounds more meaningful and easier to understand, even if you can’t hear them.

## Why This Matters  
People who are deaf or hard of hearing—especially parents or pet owners—often miss important sounds and their emotional meanings.  
- A baby’s cry might mean pain, hunger, or tiredness.  
- A dog’s bark could mean fear or excitement.  

Most apps simply label sounds like “crying” or “barking” without explaining what’s really going on. HearEmotions fills this gap.

## How It Works  
1. **Sound Classification**  
   The app listens to short audio clips and uses machine learning models to classify the sound type.  
   - **Baby sounds:** hunger, pain, tiredness  
   - **Dog sounds:** barking, growling, whimpering  

2. **Explanation with Gemma 3n**  
   After recognizing the sound, the result is sent to Gemma 3n (a local language model) to generate a short, simple explanation like a human would say.  
   Example prompt:  
   > “You are an assistant for deaf people. A dog bark with emotion ‘fear’ was detected. Explain what this means.”  
   Example response:  
   > “Your dog is barking. It seems scared.”  

   The explanations are short, friendly, and easy to understand. To make it more accessible, two emojis are added to symbolically represent the message.

## Datasets Used  
HearEmotions focuses on babies, dogs, and cats because their sounds (crying, barking, meowing) carry emotional meaning relevant for caregivers, especially those who are deaf or hard of hearing. These sounds express needs such as hunger, discomfort, or anxiety.  

- **Baby Crying Sounds Dataset**  
  - ~280 WAV files  
  - Sample rate: 16 kHz  
  - Duration: 3–5 seconds  
  - Labels: hunger, pain, tiredness, diaper, fear, etc.  

- **Dog Voice Emotion Dataset**  
  - ~450 WAV files  
  - Sample rate: 16 kHz  
  - Duration: ~3 seconds  
  - Labels: angry, fearful, happy, neutral, sad  

These public datasets from Kaggle were used to train the classifiers.

## Behind the Scenes  

| Part             | What It Does                              |  
|------------------|------------------------------------------|  
| Audio Input      | Upload WAV file or (future) microphone input |  
| Feature Extraction | Extracts MFCC features from audio        |  
| Classifier       | Random Forest model classifies the sound |  
| Gemma 3n         | Generates a simple, friendly explanation |  
| Output           | Displays sound type, emoji, confidence, and explanation |  

All processing happens locally—no audio or data is sent to the cloud.

## Privacy & Accessibility  
- 100% offline processing  
- No audio is saved or uploaded anywhere  
- Friendly, easy-to-understand explanations  
- Accessible for users of all ages and abilities  

## What Makes HearEmotions Special?  
- Combines sound recognition with natural language explanations  
- Uses real baby and animal recordings  
- Designed specifically for deaf and hard-of-hearing users  
- Works fully offline—no internet required  
- Uses emojis and clear sentences for better understanding  

## Code & API Notes  
By default, HearEmotions runs fully offline using a local Gemma 3n model (via GGUF + ctransformers).  
API keys have been removed from the demo code to protect privacy. To enable API usage for testing:  
1. Add your API key securely (e.g., via Kaggle secrets or environment variables)  
2. Uncomment the relevant code sections for API calls  

## Real Impact Examples  
- Deaf parents understand why their baby is crying  
- Pet owners know if their dog feels scared or excited  
- Caregivers receive emotional context, not just noise alerts  

  
  
