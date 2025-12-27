const safeBtoa = (str: string) => btoa(unescape(encodeURIComponent(str)));

// Refined SVG Logos with more detail
const svg3gpp = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 180">
  <g fill="none" stroke-width="16">
    <path d="M 50 60 Q 130 15 210 60" stroke="#009639" />
    <path d="M 50 88 Q 130 43 210 88" stroke="#D30F8B" />
    <path d="M 50 116 Q 130 71 210 116" stroke="#065FA3" />
  </g>
  <text x="130" y="158" font-family="Arial, sans-serif" font-weight="900" font-size="44" text-anchor="middle" fill="#333">3GPP</text>
</svg>`;

const svgOran = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 80">
  <text x="10" y="55" font-family="Arial, sans-serif" font-weight="900" font-size="48" fill="#E60028">O-RAN</text>
</svg>`;

const svgFapi = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect width="100" height="100" rx="20" fill="#CC0000"/>
  <text x="50" y="65" font-family="Arial, sans-serif" font-weight="900" font-size="38" fill="#FFF" text-anchor="middle">scf</text>
</svg>`;

const svgLangChain = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <path d="M50 20 L80 50 L50 80 L20 50 Z" fill="none" stroke="#1C3C3C" stroke-width="8"/>
  <circle cx="50" cy="50" r="10" fill="#1C3C3C"/>
  <path d="M50 20 Q80 20 80 50 Q80 80 50 80 Q20 80 20 50 Q20 20 50 20" fill="none" stroke="#1C3C3C" stroke-width="4" stroke-dasharray="4 2"/>
</svg>`;

const svgOpenAI = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <path fill="#111" d="M85.3 46.2c.4-2.8.1-5.6-1-8.2-1.8-4.6-5.5-8.2-10.1-10-2.6-1.1-5.4-1.4-8.2-1-1.3-3.8-3.7-7-7-9.3-5.3-3.7-12.1-4.2-17.9-1.3-3.3 1.6-6.1 4.2-8 7.3-3.8-1.3-7.8-1.5-11.7-.6-5.8 1.4-10.7 5.5-13 11-1.3 3.1-1.8 6.4-1.3 9.7-3.8 1.3-7 3.7-9.3 7-3.7 5.3-4.2 12.1-1.3 17.9 1.6 3.3 4.2 6.1 7.3 8-1.3 3.8-1.5 7.8-.6 11.7 1.4 5.8 5.5 10.7 11 13 3.1 1.3 6.4 1.8 9.7 1.3 1.3 3.8 3.7 7 7 9.3 5.3 3.7 12.1 4.2 17.9 1.3 3.3-1.6 6.1-4.2 8-7.3 3.8 1.3 7.8 1.5 11.7.6 5.8-1.4 10.7-5.5 13-11 1.3-3.1 1.8-6.4 1.3-9.7 3.8-1.3 7-3.7 9.3-7 3.7-5.3 4.2-12.1 1.3-17.9-1.7-3.3-4.3-6.1-7.4-8zm-41 38c-3.2 1.8-7 2.1-10.3 1.1-.9-.3-1.8-.7-2.6-1.2l14.7-8.5c.3-.2.5-.5.5-.8v-21l7.8 4.5v17.4c0 .3-.2.6-.5.7l-9.6 7.8zm-29.3-10.8c-1.8-3.2-2.1-7-1.1-10.3.3-.9.7-1.8 1.2-2.6l14.7 8.5c.3.2.6.2.8.2s.5-.1.7-.4l18.2-31.5 7.8 4.5-18.2 31.5c-.2.3-.2.6-.2.8s.1.5.4.7l15.1 8.7c.3.2.6.2.8.2s.5-.1.7-.4l9.6-16.6 7.8 4.5-9.6 16.6c-.3.5-.8.8-1.4.8h-11.1c-.6 0-1.1-.3-1.4-.8l-15.1-8.7c-.3-.2-.5-.5-.5-.8v-11.1c0-.6.3-1.1.8-1.4l15.1-8.7c.3-.2.6-.2.8-.2s.5.1.7.4l9.6 16.6 7.8-4.5-9.6-16.6c-.3-.5-.3-1.1 0-1.6l11.1-19.2c.3-.5.8-.8 1.4-.8h22.2c.6 0 1.1.3 1.4.8l11.1 19.2c.3.5.3 1.1 0 1.6l-11.1 19.2c-.3.5-.8.8-1.4.8h-11.1c-.6 0-1.1-.3-1.4-.8l-15.1-8.7c-.3-.2-.6-.2-.8-.2s-.5.1-.7.4l-9.6 16.6-7.8-4.5 9.6-16.6c.3-.5.3-1.1 0-1.6l-11.1-19.2c-.3-.5-.8-.8-1.4-.8H31.5c-.6 0-1.1.3-1.4.8l-11.1 19.2c-.3.5-.3 1.1 0 1.6l11.1 19.2c.3.5.8.8 1.4.8h11.1c.6 0 1.1-.3 1.4-.8l15.1 8.7c.3.2.6.2.8.2s.5-.1.7-.4l9.6-16.6 7.8 4.5-9.6 16.6c-.3.5-.8.8-1.4.8h-11.1c-.6 0-1.1-.3-1.4-.8l-15.1-8.7c-.3-.2-.5-.5-.5-.8V72.2c0-.6.3-1.1.8-1.4l15.1-8.7c.3-.2.6-.2.8-.2s.5.1.7.4l9.6 16.6 7.8-4.5-9.6-16.6c-.3-.5-.3-1.1 0-1.6l11.1-19.2c.3-.5.8-.8 1.4-.8h22.2c.6 0 1.1.3 1.4.8l11.1 19.2c.3.5.3 1.1 0 1.6L68.5 61c-.3.5-.8.8-1.4.8H56c-.6 0-1.1-.3-1.4-.8l-15.1-8.7c-.3-.2-.6-.2-.8-.2s-.5.1-.7.4L28.4 69.2l-7.8-4.5 9.6-16.6c.3-.5.3-1.1 0-1.6l-11.1-19.2c-.3-.5-.8-.8-1.4-.8H6.5c-.6 0-1.1.3-1.4.8L-6 46.5c-.3.5-.3 1.1 0 1.6l11.1 19.2c.3.5.8.8 1.4.8h11.1c.6 0 1.1-.3 1.4-.8l15.1 8.7c.3.2.6.2.8.2s.5-.1.7-.4l9.6-16.6 7.8 4.5-9.6 16.6c-.3.5-.8.8-1.4.8h-11.1c-.6 0-1.1-.3-1.4-.8l-15.1-8.7c-.3-.2-.5-.5-.5-.8V40.2c0-.6.3-1.1.8-1.4l15.1-8.7c.3-.2.6-.2.8-.2s.5.1.7.4l9.6 16.6 7.8-4.5-9.6-16.6c-.3-.5-.3-1.1 0-1.6l11.1-19.2c.3-.5.8-.8 1.4-.8h22.2c.6 0 1.1.3 1.4.8l11.1 19.2c.3.5.3 1.1 0 1.6l-11.1 19.2c-.3.5-.8.8-1.4.8h-11.1c-.6 0-1.1-.3-1.4-.8l-15.1-8.7c-.3-.2-.6-.2-.8-.2s-.5.1-.7.4l-9.6 16.6-7.8-4.5 9.6-16.6c.3-.5.3-1.1 0-1.6l-11.1-19.2c-.3-.5-.8-.8-1.4-.8H31.5c-.6 0-1.1.3-1.4.8L19 40.2c-.3.5-.3 1.1 0 1.6l11.1 19.2c.3.5.8.8 1.4.8h11.1c.6 0 1.1-.3 1.4-.8l15.1 8.7c.3.2.6.2.8.2s.5-.1.7-.4l9.6-16.6 7.8 4.5-9.6 16.6c-.3.5-.8.8-1.4.8h-11.1c-.6 0-1.1-.3-1.4-.8l-15.1-8.7c-.3-.2-.5-.5-.5-.8V8.2c0-.6.3-1.1.8-1.4l15.1-8.7c.3-.2.6-.2.8-.2s.5.1.7.4l9.6 16.6 7.8-4.5-9.6-16.6c-.3-.5-.3-1.1 0-1.6L59 8.2c.3-.5.8-.8 1.4-.8h22.2c.6 0 1.1.3 1.4.8l11.1 19.2c.3.5.3 1.1 0 1.6l-11.1 19.2c-.3.5-.8.8-1.4.8H82.6c-.6 0-1.1-.3-1.4-.8l-15.1-8.7c-.3-.2-.6-.2-.8-.2s-.5.1-.7.4l-9.6 16.6-7.8-4.5 9.6-16.6c.3-.5.3-1.1 0-1.6l-11.1-19.2c-.3-.5-.8-.8-1.4-.8H42.5c-.6 0-1.1.3-1.4.8l-11.1 19.2c-.3.5-.3 1.1 0 1.6l11.1 19.2c.3.5.8.8 1.4.8h11.1c.6 0 1.1-.3 1.4-.8l15.1 8.7c.3.2.6.2.8.2s.5-.1.7-.4l9.6-16.6 7.8 4.5-9.6 16.6c-.3.5-.8.8-1.4.8h-11.1c-.6 0-1.1-.3-1.4-.8l-15.1-8.7c-.3-.2-.5-.5-.5-.8V72.2z"/>
</svg>`;

const svgLlama = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <path d="M30 80 Q30 40 50 30 Q70 40 70 80" fill="#0668E1" stroke="#004182" stroke-width="2"/>
  <circle cx="40" cy="45" r="3" fill="white"/>
  <circle cx="60" cy="45" r="3" fill="white"/>
  <rect x="45" y="80" width="10" height="15" fill="#004182"/>
</svg>`;

const svgChroma = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <path d="M20 50 A30 15 0 1 0 80 50 A30 15 0 1 0 20 50" fill="#FF8C00" opacity="0.8"/>
  <path d="M20 60 A30 15 0 1 0 80 60 L80 50 A30 15 0 1 1 20 50 Z" fill="#E67E22"/>
  <path d="M20 70 A30 15 0 1 0 80 70 L80 60 A30 15 0 1 1 20 60 Z" fill="#D35400"/>
</svg>`;

const svgNeo4j = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="30" cy="30" r="10" fill="#018BFF"/>
  <circle cx="70" cy="30" r="10" fill="#018BFF"/>
  <circle cx="50" cy="70" r="15" fill="#018BFF"/>
  <line x1="30" y1="30" x2="50" y2="70" stroke="#018BFF" stroke-width="4"/>
  <line x1="70" y1="30" x2="50" y2="70" stroke="#018BFF" stroke-width="4"/>
  <text x="50" y="75" font-family="Arial, sans-serif" font-weight="900" font-size="12" fill="white" text-anchor="middle">N</text>
</svg>`;

const svgReranker = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="20" y="20" width="60" height="10" rx="2" fill="#F59E0B" opacity="0.3"/>
  <rect x="20" y="40" width="60" height="10" rx="2" fill="#F59E0B" opacity="0.6"/>
  <rect x="20" y="60" width="60" height="10" rx="2" fill="#F59E0B"/>
  <path d="M10 40 L15 50 L10 60" fill="none" stroke="#F59E0B" stroke-width="4"/>
</svg>`;

const svgGear = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="25" fill="none" stroke="#db2777" stroke-width="8"/>
  <path d="M50 10 L50 25 M50 75 L50 90 M10 50 L25 50 M75 50 L90 50 M22 22 L32 32 M68 68 L78 78 M22 78 L32 68 M68 22 L78 32" stroke="#db2777" stroke-width="8" stroke-linecap="round"/>
</svg>`;

const svgCompass = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="40" fill="none" stroke="#be123c" stroke-width="4"/>
  <path d="M50 20 L55 50 L50 80 L45 50 Z" fill="#be123c"/>
</svg>`;

const svgShield = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <path d="M50 10 C30 15 15 25 15 45 C15 70 50 90 50 90 C50 90 85 70 85 45 C85 25 70 15 50 10" fill="#e11d48"/>
  <path d="M50 25 L50 75 M35 45 L65 45" stroke="white" stroke-width="6" stroke-linecap="round"/>
</svg>`;

const svgCache = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="15" y="15" width="70" height="20" rx="4" fill="#f97316"/>
  <rect x="15" y="40" width="70" height="20" rx="4" fill="#f97316" opacity="0.7"/>
  <rect x="15" y="65" width="70" height="20" rx="4" fill="#f97316" opacity="0.4"/>
  <path d="M50 10 L50 90" stroke="white" stroke-width="2" stroke-dasharray="2 4"/>
</svg>`;

const svgUser = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="35" r="20" fill="#1e293b"/>
  <path d="M20 85 Q20 60 50 60 Q80 60 80 85" fill="#1e293b"/>
</svg>`;

const svgWeights = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="20" y="20" width="60" height="60" rx="8" fill="#e5e7eb" stroke="#94a3b8" stroke-width="2"/>
  <line x1="30" y1="40" x2="70" y2="40" stroke="#94a3b8" stroke-width="2"/>
  <line x1="30" y1="50" x2="70" y2="50" stroke="#94a3b8" stroke-width="2"/>
  <line x1="30" y1="60" x2="50" y2="60" stroke="#94a3b8" stroke-width="2"/>
</svg>`;

const svgBook = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <path d="M20 20 H70 V80 H20 Z" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
  <path d="M70 20 L80 30 V90 L70 80 Z" fill="#fde68a" stroke="#d97706" stroke-width="2"/>
  <path d="M20 80 L30 90 H80 L70 80 Z" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
</svg>`;

export const LOGO_3GPP = `data:image/svg+xml;base64,${safeBtoa(svg3gpp)}`;
export const LOGO_ORAN = `data:image/svg+xml;base64,${safeBtoa(svgOran)}`;
export const LOGO_FAPI = `data:image/svg+xml;base64,${safeBtoa(svgFapi)}`;
export const LOGO_LANGCHAIN = `data:image/svg+xml;base64,${safeBtoa(svgLangChain)}`;
export const LOGO_OPENAI = `data:image/svg+xml;base64,${safeBtoa(svgOpenAI)}`;
export const LOGO_LLAMA = `data:image/svg+xml;base64,${safeBtoa(svgLlama)}`;
export const LOGO_CHROMA = `data:image/svg+xml;base64,${safeBtoa(svgChroma)}`;
export const LOGO_NEO4J = `data:image/svg+xml;base64,${safeBtoa(svgNeo4j)}`;
export const LOGO_RERANKER = `data:image/svg+xml;base64,${safeBtoa(svgReranker)}`;
export const LOGO_USER = `data:image/svg+xml;base64,${safeBtoa(svgUser)}`;
export const LOGO_WEIGHTS = `data:image/svg+xml;base64,${safeBtoa(svgWeights)}`;
export const LOGO_GEAR = `data:image/svg+xml;base64,${safeBtoa(svgGear)}`;
export const LOGO_SHIELD = `data:image/svg+xml;base64,${safeBtoa(svgShield)}`;
export const LOGO_COMPASS = `data:image/svg+xml;base64,${safeBtoa(svgCompass)}`;
export const LOGO_CACHE = `data:image/svg+xml;base64,${safeBtoa(svgCache)}`;
export const LOGO_BOOK = `data:image/svg+xml;base64,${safeBtoa(svgBook)}`;