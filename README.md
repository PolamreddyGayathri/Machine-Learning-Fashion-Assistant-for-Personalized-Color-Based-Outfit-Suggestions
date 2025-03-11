# 👗 Machine Learning Fashion Assistant  
### Personalized Color-Based Outfit Suggestions  

##  Overview  
This project is a **fashion assistant** that detects **skin tone and eye color** from user-uploaded images and recommends **personalized outfit colors** based on detected features. It also allows users to **save favorite outfits** and interact with a **user-friendly web interface**.

##  Features  
1️. **Skin Tone & Eye Color Detection**: Uses **Deep Learning & OpenCV** to analyze facial features.  
2️. **Color Palette Generation**: Uses **K-Means clustering** to extract a **personalized color palette**.  
3️. **Outfit Recommendation System**: Uses **Content-Based & Hybrid Filtering** for recommendations.  
4️. **Wishlist Feature**: Users can **save favorite outfits** for future reference.  
5️. **Web-Based Interface**: Developed using **React.js (Frontend)** and **FastAPI (Backend)**.  

---

##  Installation Guide  

### **Clone the Repository**  
```sh
git clone (https://github.com/PolamreddyGayathri/Machine-Learning-Fashion-Assistant-for-Personalized-Color-Based-Outfit-Suggestions.git)
## **Frontend**  
cd color_analysis
cd frontend 
npm run start 

## **Backend**
cd backend 
npm start
## **Run**
python -m uvicorn app:app —reload

