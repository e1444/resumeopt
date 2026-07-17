export function Spinner({ label }: { label?: string }) {
  return (
    <span className="spinner-row">
      <span className="spinner" role="status" aria-label={label ?? 'Loading'} />
      {label && <span>{label}</span>}
    </span>
  );
}
