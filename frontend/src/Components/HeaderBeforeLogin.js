import React from "react";
import { Link } from "react-router-dom";

const HeaderBeforeLogin = () => {
  return (
    <header className="p-4 bg-gray-900 text-white flex justify-between">
      <h1 className="text-xl font-bold">My App</h1>
      <nav>
        <Link to="/login" className="px-4 py-2 bg-blue-500 rounded">Login</Link>
      </nav>
    </header>
  );
};

export default HeaderBeforeLogin;
