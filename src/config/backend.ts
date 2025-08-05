
export const getBackendUrl = (route: string): string => {
  const baseUrl = import.meta.env.VITE_BACKEND_URL || 'https://184f-34-44-17-76.ngrok-free.app';
  return `${baseUrl}${route}`;
};
