# Security

Never commit `backend/.env` or place a real credential in `.env.example`.

For a production deployment:

1. Set a strong `ADMIN_API_KEY` and enter it through the UI only when managing documents.
2. Serve the frontend and API over TLS.
3. Put user authentication and authorization in front of chat history if multiple users share the deployment.
4. Treat uploaded documents as untrusted input and restrict who can manage the knowledge base.
5. Rotate any key that has appeared in a source archive, Git commit, log, screenshot or Docker image.

The `X-Admin-Key` mechanism protects document-management operations but is not a replacement for a complete identity system.
