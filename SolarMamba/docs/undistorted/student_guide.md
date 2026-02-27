# The Student's Guide to SolarMamba

Welcome! If you know some Python but are new to Deep Learning (DL) and Artificial Intelligence (AI), this guide is for you. We will walk through how **SolarMamba** works, from the moment we load a picture of the sky to the moment we predict how much sunlight will hit a solar panel.

---

## Part 1: The Big Picture

### What are we trying to do?
Imagine you are running a solar power plant. You need to know **exactly how much sunlight** will hit your panels in the next few minutes so you can tell the power grid what to expect.
*   **The Problem:** Clouds move fast. If a cloud covers the sun, power drops instantly.
*   **The Solution:** We use a camera looking up at the sky (an **All-Sky Imager**) and weather data to predict the future sunlight.

### How does our "Brain" (The Model) work?
Our model, **SolarMamba**, works like a human meteorologist:
1.  **It Looks (Visual):** It sees the clouds in the image.
2.  **It Feels (Temporal):** It looks at the history of temperature and pressure.
3.  **It Knows Physics:** It knows where the sun is in the sky (using math, not guessing).

---

## Part 2: The Data (Feeding the Brain)

Before the model can learn, we need to prepare the food (data).

### 1. The Image (The Eyes)
We take a picture of the sky.
*   **Resize:** We shrink it to 512x512 pixels so the computer can process it faster.
*   **Masking:** The camera sees the ground and horizon (trees, buildings). We don't care about those. We draw a black circle around the center so the model only focuses on the sky.

### 2. The Weather (The Memory)
We look at the past 60 minutes of weather data. But we don't just give it raw numbers; we give it **Physics-Informed** numbers.
*   **Solar Zenith Angle (SZA):** How high is the sun? (0° = overhead, 90° = horizon).
*   **Azimuth:** Which direction is the sun? (North, South, East, West).
*   **Clear Sky Index ($k^*$):** This is a clever trick. Instead of predicting "500 Watts of energy," we predict "50% of the maximum possible energy."
    *   If $k^* = 1.0$, it's a sunny day.
    *   If $k^* = 0.2$, it's very cloudy.
    *   *Why?* Because predicting a percentage is easier for AI than predicting a raw number that changes wildly from sunrise to sunset.

---

## Part 3: The Architecture (The Brain Structure)

Our model has three main parts working together.

### 1. The Visual Encoder (MambaVision)
Think of this as the part of the brain that processes vision.
*   **What it does:** It looks at the image and extracts "features."
*   **Stages:** It looks at the image in 4 stages, going from small details to big patterns.
    *   *Stage 1:* "I see edges and lines."
    *   *Stage 2:* "I see shapes like puffs of clouds."
    *   *Stage 3:* "I see a large cloud formation."
    *   *Stage 4:* "I understand the cloud is moving towards the sun."

### 2. The Temporal Encoder (Pyramid TCN)
Think of this as the part of the brain that understands time and rhythm.
*   **What it does:** It looks at the weather history (past 60 mins).
*   **Pyramid:** It uses 4 different "branches" to look at time differently.
    *   *Branch 1:* Looks at minute-by-minute changes (Noise).
    *   *Branch 2:* Looks at 5-minute trends.
    *   *Branch 3:* Looks at 15-minute trends.
    *   *Branch 4:* Looks at the whole hour (Trend).

### 3. Ladder Fusion (The Connection)
This is where the magic happens. We connect the "Vision" part to the "Time" part.
*   **Gating:** Imagine the Time part telling the Vision part what to focus on.
    *   *Example:* If the weather data says "It's getting colder fast," the Time part tells the Vision part, "Hey, look closely for storm clouds!"
*   **Math:** $Output = Image + (Image \times Gate)$. The "Gate" is a filter created by the weather data.

---

## Part 4: Training (Teaching the Brain)

Now we have a brain, but it knows nothing. We need to teach it.

### 1. The Loss Function (The Scoreboard)
How do we tell the model if it did a good job? We use a **Loss Function**.
*   **MSE (Mean Squared Error):** We compare the model's prediction to the real answer.
    *   *Prediction:* 0.8
    *   *Reality:* 1.0
    *   *Error:* $(0.8 - 1.0)^2 = 0.04$.
*   **Goal:** The model tries to make this number as close to 0 as possible.

### 2. The Optimizer (The Teacher)
The **Optimizer** is the algorithm that adjusts the model's brain cells (weights) to reduce the error.
*   **AdamW:** This is a very popular, smart teacher. It adjusts the learning speed automatically. If the model is learning fast, it speeds up. If it's stuck, it slows down and looks carefully.

### 3. The Scheduler (The Schedule)
We don't want to teach at the same speed forever.
*   **Cosine Annealing:** We start with a high learning rate (learning fast), then slow down to refine the details, then speed up again to jump out of bad habits. It's like studying hard, taking a break, then studying hard again.

---

## Part 5: Inference (Using the Brain)

Once the model is trained, we use it for **Inference** (Prediction). We can predict for different future times (Horizons), like 1 minute, 5 minutes, or 15 minutes ahead.

1.  **Input:** We give it a *new* image and the last 60 minutes of weather.
2.  **Process:**
    *   The Visual Encoder sees the clouds.
    *   The Temporal Encoder feels the trends.
    *   The Fusion layer combines them.
3.  **Output:** The model spits out a number, say `0.75`.
4.  **Reconstruction:**
    *   The model says: "The sky is 75% clear." ($k^* = 0.75$)
    *   Physics says: "If the sky was perfectly clear, you'd get 1000 Watts." ($GHI_{cs} = 1000$)
    *   **Final Prediction:** $0.75 \times 1000 = 750$ Watts.

### Evaluating Performance
We also check "who did the work?" by covering the model's eyes (Image-Blind) or erasing its memory (Time-Blind) to see which part is more important for different forecast times. Usually, memory helps for short term (1 min), and vision helps for longer term (15 min).

---

## Summary Checklist for Students

- [ ] **Data:** Images + Weather History.
- [ ] **Preprocessing:** Mask images, calculate Sun position (Physics).
- [ ] **Model:** MambaVision (Eyes) + TCN (Memory) + Fusion (Logic).
- [ ] **Training:** Guess -> Check Error (MSE) -> Adjust Weights (AdamW).
- [ ] **Result:** A highly accurate prediction of solar energy!

Good luck on your AI journey!
