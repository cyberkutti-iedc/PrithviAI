# PrithviAI Frontend

## Overview

The **PrithviAI** frontend is a sleek and modern web interface designed to engage users with interactive visualizations and insights into global climate data. Featuring a stylish dark theme inspired by space and solar system elements, this landing page brings a sophisticated user experience to life. Built with modern web technologies, it allows users to explore various environmental data with ease.

## Features

- **Elegant Landing Page:** A visually stunning landing page with interactive components.
- **Dark Mode Design:** Solar system and space-inspired dark theme for a unique look and feel.
- **Responsive UI:** Built with Tailwind CSS to ensure an optimal experience across devices.
- **Data Visualization:** Displays emissions, climate data, and AI model predictions in a user-friendly way.
- **Interactive Climate Story:** Tells a dynamic climate story based on AI-driven insights.

## Tech Stack

### Frontend
- **Next.js**: A React-based framework for server-side rendering and static site generation.
- **Tailwind CSS**: A utility-first CSS framework for rapidly building custom designs.
  
### Hosting
- **Vercel**: The platform used for hosting and deploying the frontend of PrithviAI.

## Getting Started

### Prerequisites

To get started with the frontend, ensure you have the following installed:

- **Node.js** (version 14.x or higher)
- **NPM** (comes with Node.js) or **Yarn** (optional but recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cyberkutti-iedc/PrithviAI.git
   cd PrithviAI/frontend
   ```

2. **Install dependencies:**
   If you are using npm:
   ```bash
   npm install
   ```

   If you are using yarn:
   ```bash
   yarn install
   ```

### Running the Frontend Locally

1. **Start the development server:**
   ```bash
   npm run dev
   ```

2. Open your browser and navigate to `http://localhost:3000` to view the frontend.

### Building for Production

1. **Create a production build:**
   ```bash
   npm run build
   ```

2. **Start the production server:**
   ```bash
   npm run start
   ```

### Environment Variables

The following environment variables should be set for the project:

- **NEXT_PUBLIC_API_URL**: The base URL of the backend API that provides emission and climate data.

Example:
```bash
NEXT_PUBLIC_API_URL=https://api.prithviai.com
```

### Folder Structure

- **pages/**: Contains the Next.js pages (routes).
- **components/**: Reusable components across the frontend.
- **styles/**: Contains Tailwind CSS configurations and custom styles.
- **public/**: Static assets like images, icons, and fonts.

### Deployment

This project is deployed using **Vercel**. To deploy, follow these steps:

1. Commit and push your changes to the main branch:
   ```bash
   git add .
   git commit -m "Deploying updates"
   git push origin main
   ```

2. Vercel will automatically detect your changes and deploy them to the production URL.

### Project Dependencies

Here is a list of key dependencies used in this project:

- **Next.js**: `^12.x`
- **Tailwind CSS**: `^2.x`

You can find the complete list in the `package.json` file.



---

**PrithviAI Team**  
Delivering insights for a better planet.
