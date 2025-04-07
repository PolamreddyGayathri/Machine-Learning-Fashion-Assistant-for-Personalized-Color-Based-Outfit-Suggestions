// const express = require("express");
// const connectDB = require("./config/db");
// const userRoutes = require("./routes/UserRoutes");
// const bodyParser = require("body-parser");
// const cors = require("cors");
// const app = express();
// connectDB();
// app.use(cors());
// app.use(bodyParser.urlencoded({ extended: false }));
// app.use(bodyParser.json());

// app.use("/user", userRoutes);

// app.listen(8080, () => {
//   console.log("Server running");
// });
const express = require("express");
const connectDB = require("./config/db");
const userRoutes = require("./routes/UserRoutes");
const bodyParser = require("body-parser");
const cors = require("cors");
const mongoose = require("mongoose");
const dotenv = require("dotenv");
const app = express();
connectDB();
app.use(cors());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

dotenv.config();
const MONGO_URI = process.env.MONGO_URI;
mongoose
  .connect(MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log("✅ MongoDB Connected!"))
  .catch((err) => console.error("❌ MongoDB Connection Error:", err));
app.use("/user", userRoutes);

// app.listen(8080, () => {
//   console.log("Server running");
// });
const PORT = process.env.PORT || 8080; // Use a dynamic port
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
