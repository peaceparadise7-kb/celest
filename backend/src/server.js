import app from "./app.js";

const PORT = process.env.PORT || 5000;

const server = app.listen(PORT, () => {
  console.log(
    `🚀 Celest Backend running in ${process.env.NODE_ENV || "development"} mode on port ${PORT}`
  );
});

process.on("unhandledRejection", (err) => {
  console.error(
    `🔴 Shutting down server due to Unhandled Rejection: ${err.message}`
  );

  server.close(() => process.exit(1));
});