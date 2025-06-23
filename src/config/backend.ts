
export const getBackendUrl = (route: string): string => {
  const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
  return `${baseUrl}${route}`;
};
