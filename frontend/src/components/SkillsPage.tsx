import { useEffect, useMemo, useState } from 'react';
import { api, type Skill } from '../api';
import { ConfirmDialog } from './ConfirmDialog';
import { Spinner } from './Spinner';

type SortOrder = 'name-asc' | 'name-desc' | 'aliases-desc';

export function SkillsPage() {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [newName, setNewName] = useState('');
  const [newAliases, setNewAliases] = useState('');
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState('');
  const [sortOrder, setSortOrder] = useState<SortOrder>('name-asc');
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [editingName, setEditingName] = useState<string | null>(null);
  const [editAliasesDraft, setEditAliasesDraft] = useState('');
  const [savingEdit, setSavingEdit] = useState(false);
  const [togglingAlwaysInclude, setTogglingAlwaysInclude] = useState<string | null>(null);

  const reload = async () => {
    setLoading(true);
    setError(null);
    try {
      setSkills(await api.listSkills());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    reload();
  }, []);

  useEffect(() => {
    if (!success) return;
    const timer = window.setTimeout(() => setSuccess(null), 3000);
    return () => window.clearTimeout(timer);
  }, [success]);

  const handleAdd = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!newName.trim()) return;
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const aliases = newAliases
        .split(',')
        .map((alias) => alias.trim())
        .filter(Boolean);
      setSkills(await api.addSkill(newName.trim(), aliases));
      setSuccess(`Added "${newName.trim()}" to the cache.`);
      setNewName('');
      setNewAliases('');
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (name: string) => {
    setPendingDelete(null);
    setError(null);
    setSuccess(null);
    try {
      setSkills(await api.deleteSkill(name));
      setSuccess(`Removed "${name}" from the cache.`);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const startEditingAliases = (skill: Skill) => {
    setEditingName(skill.name);
    setEditAliasesDraft(skill.aliases.join(', '));
    setError(null);
  };

  const cancelEditingAliases = () => {
    setEditingName(null);
    setEditAliasesDraft('');
  };

  const handleSaveAliases = async (name: string) => {
    setSavingEdit(true);
    setError(null);
    setSuccess(null);
    try {
      const aliases = editAliasesDraft
        .split(',')
        .map((alias) => alias.trim())
        .filter(Boolean);
      setSkills(await api.updateSkill(name, { aliases }));
      setSuccess(`Updated aliases for "${name}".`);
      setEditingName(null);
      setEditAliasesDraft('');
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSavingEdit(false);
    }
  };

  const handleToggleAlwaysInclude = async (name: string, alwaysInclude: boolean) => {
    setTogglingAlwaysInclude(name);
    setError(null);
    try {
      setSkills(await api.updateSkill(name, { always_include: alwaysInclude }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setTogglingAlwaysInclude(null);
    }
  };

  const visibleSkills = useMemo(() => {
    const query = search.trim().toLowerCase();
    const filtered = query
      ? skills.filter(
          (skill) =>
            skill.name.toLowerCase().includes(query) ||
            skill.aliases.some((alias) => alias.toLowerCase().includes(query))
        )
      : skills;

    const sorted = [...filtered];
    switch (sortOrder) {
      case 'name-asc':
        sorted.sort((a, b) => a.name.localeCompare(b.name));
        break;
      case 'name-desc':
        sorted.sort((a, b) => b.name.localeCompare(a.name));
        break;
      case 'aliases-desc':
        sorted.sort((a, b) => b.aliases.length - a.aliases.length);
        break;
    }
    return sorted;
  }, [skills, search, sortOrder]);

  return (
    <div className="page">
      <h2>Skills cache</h2>
      <p className="hint">
        Canonical skills matched against job postings. Aliases are true spelling/naming
        variants only (e.g. "torch" for "pytorch") - click a skill's aliases to edit them.
        Check "Always include" for a skill you want on every tailored resume regardless of the
        posting (e.g. a language or practice you're always comfortable listing).
      </p>

      {error && <div className="banner error">{error}</div>}
      {success && <div className="banner success">{success}</div>}

      <form className="inline-form" onSubmit={handleAdd}>
        <input
          type="text"
          placeholder="Skill name (e.g. kubernetes)"
          value={newName}
          onChange={(event) => setNewName(event.target.value)}
        />
        <input
          type="text"
          placeholder="Aliases, comma-separated (optional)"
          value={newAliases}
          onChange={(event) => setNewAliases(event.target.value)}
        />
        <button type="submit" disabled={saving || !newName.trim()}>
          {saving ? 'Adding…' : 'Add skill'}
        </button>
      </form>

      <div className="filter-toolbar">
        <input
          type="text"
          placeholder="Search by name or alias…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <select value={sortOrder} onChange={(event) => setSortOrder(event.target.value as SortOrder)}>
          <option value="name-asc">Name (A-Z)</option>
          <option value="name-desc">Name (Z-A)</option>
          <option value="aliases-desc">Most aliases first</option>
        </select>
      </div>

      {loading ? (
        <Spinner label="Loading skills…" />
      ) : (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Aliases</th>
                <th>Always include</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {visibleSkills.map((skill) => (
                <tr key={skill.name}>
                  <td>{skill.name}</td>
                  <td className="wrap">
                    {editingName === skill.name ? (
                      <div className="inline-edit">
                        <input
                          type="text"
                          autoFocus
                          value={editAliasesDraft}
                          onChange={(event) => setEditAliasesDraft(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter') handleSaveAliases(skill.name);
                            if (event.key === 'Escape') cancelEditingAliases();
                          }}
                          placeholder="Aliases, comma-separated"
                        />
                        <button
                          className="link-button"
                          disabled={savingEdit}
                          onClick={() => handleSaveAliases(skill.name)}
                        >
                          {savingEdit ? 'Saving…' : 'Save'}
                        </button>
                        <button className="link-button" disabled={savingEdit} onClick={cancelEditingAliases}>
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        className="alias-edit-trigger"
                        onClick={() => startEditingAliases(skill)}
                        title="Click to edit aliases"
                      >
                        {skill.aliases.join(', ') || <span className="muted-text">Click to add aliases</span>}
                      </button>
                    )}
                  </td>
                  <td>
                    <input
                      type="checkbox"
                      checked={skill.always_include}
                      disabled={togglingAlwaysInclude === skill.name}
                      onChange={(event) => handleToggleAlwaysInclude(skill.name, event.target.checked)}
                      aria-label={`Always include ${skill.name}`}
                    />
                  </td>
                  <td>
                    <button
                      className="link-button danger"
                      disabled={editingName === skill.name}
                      onClick={() => setPendingDelete(skill.name)}
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
              {visibleSkills.length === 0 && (
                <tr>
                  <td colSpan={4}>
                    <em>{skills.length === 0 ? 'No skills in the cache yet.' : 'No skills match your search.'}</em>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {pendingDelete && (
        <ConfirmDialog
          title="Remove skill"
          message={`Remove "${pendingDelete}" from the skills cache? This can't be undone from the UI (a backup is kept on disk).`}
          confirmLabel="Remove"
          onConfirm={() => handleDelete(pendingDelete)}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </div>
  );
}
