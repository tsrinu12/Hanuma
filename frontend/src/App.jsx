import React from "react";
import { Link, Route, Routes } from "react-router-dom";
import Home from "./components/Home.jsx";
import Watch from "./components/Watch.jsx";
import Upload from "./components/Upload.jsx";
import Search from "./components/Search.jsx";

export default function App() {
  return (
    <div style={{ minHeight: "100vh" }}>
      <nav style={{
        display: "flex", gap: 16, padding: "12px 24px",
        borderBottom: "1px solid #222", background: "#111", alignItems: "center"
      }}>
        <Link to="/" style={{ color: "#fff", fontWeight: 700, textDecoration: "none", fontSize: 20 }}>
          distrebute<span style={{ color: "#7c5cff" }}>.com</span>
        </Link>
        <Link to="/search" style={navStyle}>Search</Link>
        <Link to="/upload" style={navStyle}>Upload</Link>
      </nav>
      <main style={{ padding: 24 }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/watch/:id" element={<Watch />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/search" element={<Search />} />
        </Routes>
      </main>
    </div>
  );
}

const navStyle = { color: "#ccc", textDecoration: "none", fontSize: 14 };
