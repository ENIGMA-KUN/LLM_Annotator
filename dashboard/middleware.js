// Basic Auth for the whole deployment.
//
// Vercel's Hobby plan does not include Password Protection; on Hobby the production
// domain is publicly reachable by default. The dashboard bundles results that are not
// published, so it must not be open. Edge Middleware runs before every request and is
// free on Hobby.
//
// Set these in Vercel: Project Settings -> Environment Variables
//   DASHBOARD_USER      e.g. lei
//   DASHBOARD_PASSWORD  a long random string, NOT reused from anywhere else
//
// Fail closed: if the variables are missing, everything is blocked rather than exposed.

export const config = {
  // Match everything except Vercel internals. Static assets are protected too,
  // because findingsData.js is compiled into the bundle and contains the results.
  matcher: ["/((?!_vercel|favicon.ico).*)"],
};

function unauthorized(message) {
  return new Response(message, {
    status: 401,
    headers: {
      "WWW-Authenticate": 'Basic realm="LLM as Judge", charset="UTF-8"',
      "Cache-Control": "no-store",
    },
  });
}

// Compare in constant time so response latency does not leak the password.
function safeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export default function middleware(request) {
  const user = process.env.DASHBOARD_USER;
  const password = process.env.DASHBOARD_PASSWORD;

  if (!user || !password) {
    return unauthorized(
      "Access is not configured. Set DASHBOARD_USER and DASHBOARD_PASSWORD in Vercel."
    );
  }

  const header = request.headers.get("authorization") || "";
  if (!header.startsWith("Basic ")) return unauthorized("Authentication required.");

  let decoded;
  try {
    decoded = atob(header.slice(6));
  } catch {
    return unauthorized("Malformed credentials.");
  }

  const sep = decoded.indexOf(":");
  if (sep < 0) return unauthorized("Malformed credentials.");

  const okUser = safeEqual(decoded.slice(0, sep), user);
  const okPass = safeEqual(decoded.slice(sep + 1), password);

  // Evaluate both before branching so a wrong username and a wrong password
  // take the same path.
  if (!(okUser && okPass)) return unauthorized("Invalid credentials.");

  return undefined; // authenticated, continue to the app
}
