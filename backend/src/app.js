import express from "express";
import cors from "cors";
import morgan from "morgan";
import dotenv from "dotenv";

// Load environment variables
dotenv.config();

// Create the Express application
const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(morgan("dev"));

// Health check route
app.get("/api/health", (req, res) => {
  res.status(200).json({
    status: "ok",
    service: "Celest Backend",
  });
});

export default app;