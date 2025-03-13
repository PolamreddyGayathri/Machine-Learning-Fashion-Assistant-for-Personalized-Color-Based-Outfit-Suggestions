import React from "react";
import { Link, useNavigate } from "react-router-dom";

const HeaderAfterLogin = ({ setIsAuthenticated }) => {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem("user");
    localStorage.removeItem("id");
    setIsAuthenticated(false); // ✅ Reset auth state
    navigate("/");
    window.location.reload(); // Refresh page
  };

  return (
    <header className="p-4 bg-gray-900 text-white flex justify-between">
      <h1 className="text-xl font-bold">My App</h1>
      <nav className="flex gap-4">
        <Link to="/home" className="px-4 py-2 bg-blue-500 rounded">Home</Link>
        <button onClick={handleLogout} className="px-4 py-2 bg-red-500 rounded">
          Logout
        </button>
      </nav>
    </header>
  );
};

export default HeaderAfterLogin;
