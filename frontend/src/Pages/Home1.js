import React from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

const Home = () => {
  return (
    <div
      className="relative h-screen flex flex-col items-center justify-center text-white text-center overflow-hidden"
      style={{
        background:
          "linear-gradient(135deg, #1a1a1a 0%, #2c3e50 30%, #8e44ad 60%, #3498db 90%)",
      }}
    >
      <motion.div
        className="relative z-10"
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
      >
        <h1 className="text-[100px] font-extrabold uppercase drop-shadow-xl tracking-widest">
          Color Insight
        </h1>
        <h2 className="text-[50px] font-medium mt-4 opacity-90">
          What's Your Color?
        </h2>
        <p className="text-xl mt-4 relative z-10 animate-fade-in">
          Discover the meaning of colors and find your perfect style.
        </p>
        
        {/* Navigation Button */}
        <motion.div
          className="mt-10"
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.5 }}
        >
          <Link
            to="/login"
            className="inline-block px-10 py-4 bg-white text-black font-semibold rounded-full shadow-lg hover:bg-gray-200 hover:scale-110 transition-all transform"
          >
            Sign In / Login
          </Link>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Home;
