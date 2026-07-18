import { useState, useEffect } from 'react';

function LandingPage({ onGetStarted, onLogin }) {
  const [styleInjected, setStyleInjected] = useState(false);

  useEffect(() => {
    if (!styleInjected) {
      const style = document.createElement('style');
      style.textContent = `

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: #F0F4F8; }
        .fade-in { animation: fadeIn 0.8s ease-out both; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
        .slide-up { animation: slideUp 0.6s ease-out both; }
        @keyframes slideUp { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: translateY(0); } }
        .float { animation: float 6s ease-in-out infinite; }
        @keyframes float { 0%,100% { transform: translateY(0px); } 50% { transform: translateY(-20px); } }
        .pulse-glow { animation: pulseGlow 2s ease-in-out infinite; }
        @keyframes pulseGlow { 0%,100% { box-shadow: 0 0 0 0 rgba(244,162,97,0.4); } 50% { box-shadow: 0 0 0 20px rgba(244,162,97,0); } }
      `;
      document.head.appendChild(style);
      setStyleInjected(true);
    }
  }, [styleInjected]);

  const navItems = [
    { label: 'Platform', href: '#platform' },
    { label: 'Results', href: '#results' },
    { label: 'Pricing', href: '#pricing' },
  ];

  const features = [
    {
      icon: '⚡',
      title: 'Autonomous Execution',
      desc: 'Your AI runs business processes 24/7 — no human hand-holding needed.',
    },
    {
      icon: '🧠',
      title: 'Strategic Intelligence',
      desc: 'Learns your business patterns and makes decisions aligned with your goals.',
    },
    {
      icon: '📊',
      title: 'Real-Time Dashboard',
      desc: 'See exactly what your AI is doing, deciding, and achieving — live.',
    },
    {
      icon: '🔗',
      title: 'Seamless Integration',
      desc: 'Connects with your existing tools, APIs, and data sources in minutes.',
    },
  ];

  const plans = [
    {
      name: 'Starter',
      price: '499',
      period: '/month',
      features: ['1 AI Agent', '5 Integrations', 'Basic Analytics', 'Email Support'],
    },
    {
      name: 'Growth',
      price: '999',
      period: '/month',
      features: ['3 AI Agents', '15 Integrations', 'Advanced Analytics', 'Priority Support', 'Custom Training'],
      popular: true,
    },
    {
      name: 'Enterprise',
      price: 'Custom',
      period: '',
      features: ['Unlimited Agents', 'Unlimited Integrations', 'Full Analytics Suite', '24/7 Dedicated Support', 'On-Premise Option', 'SLA Guarantee'],
    },
  ];

  return (
    <div style={{ minHeight: '100vh', background: '#F0F4F8', overflow: 'hidden' }}>
      {/* Navbar */}
      <nav style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 50,
        background: 'rgba(240,244,248,0.95)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(30,58,95,0.08)',
        padding: '16px 24px',
      }}>
        <div style={{
          maxWidth: 1280,
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 36,
              height: 36,
              borderRadius: 10,
              background: 'linear-gradient(135deg, #1E3A5F, #F4A261)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 18,
              fontWeight: 900,
              color: '#fff',
            }}>
              I
            </div>
            <span style={{
              fontFamily: "'Playfair Display', serif",
              fontSize: 22,
              fontWeight: 900,
              color: '#1E3A5F',
              letterSpacing: -0.5,
            }}>
              Ia que ejecuta
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 32 }}>
            {(navItems || []).map((item, idx) => (
              <a
                key={idx}
                href={item.href}
                style={{
                  color: '#1E3A5F',
                  textDecoration: 'none',
                  fontSize: 14,
                  fontWeight: 600,
                  letterSpacing: -0.2,
                  transition: 'color 0.2s',
                }}
                onMouseEnter={e => e.target.style.color = '#F4A261'}
                onMouseLeave={e => e.target.style.color = '#1E3A5F'}
              >
                {item.label}
              </a>
            ))}
            <button
              onClick={onLogin}
              style={{
                padding: '10px 20px',
                borderRadius: 10,
                border: '2px solid #1E3A5F',
                background: 'transparent',
                color: '#1E3A5F',
                fontWeight: 700,
                fontSize: 14,
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
              onMouseEnter={e => {
                e.target.style.background = '#1E3A5F';
                e.target.style.color = '#fff';
              }}
              onMouseLeave={e => {
                e.target.style.background = 'transparent';
                e.target.style.color = '#1E3A5F';
              }}
            >
              Sign in
            </button>
            <button
              onClick={onGetStarted}
              style={{
                padding: '10px 24px',
                borderRadius: 10,
                border: 'none',
                background: '#F4A261',
                color: '#1E3A5F',
                fontWeight: 700,
                fontSize: 14,
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
              onMouseEnter={e => {
                e.target.style.background = '#e8924d';
                e.target.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={e => {
                e.target.style.background = '#F4A261';
                e.target.style.transform = 'translateY(0)';
              }}
            >
              Get Started Free
            </button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section style={{
        paddingTop: 140,
        paddingBottom: 80,
        paddingLeft: 24,
        paddingRight: 24,
        maxWidth: 1280,
        margin: '0 auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        position: 'relative',
      }}>
        <div className="fade-in" style={{ maxWidth: 820 }}>
          <div style={{
            display: 'inline-block',
            padding: '8px 16px',
            borderRadius: 20,
            background: 'rgba(244,162,97,0.15)',
            color: '#F4A261',
            fontSize: 13,
            fontWeight: 700,
            letterSpacing: 1,
            textTransform: 'uppercase',
            marginBottom: 24,
          }}>
            AI That Actually Works For Your Business
          </div>
          <h1 style={{
            fontFamily: "'Playfair Display', serif",
            fontSize: 'clamp(42px, 8vw, 72px)',
            fontWeight: 900,
            color: '#1E3A5F',
            lineHeight: 1.08,
            letterSpacing: -2,
            marginBottom: 24,
          }}>
            Your Company's AI
            <br />
            <span style={{ color: '#F4A261' }}>That Executes</span>
          </h1>
          <p style={{
            fontSize: 18,
            color: '#5a6b7e',
            lineHeight: 1.6,
            maxWidth: 600,
            margin: '0 auto 36px',
          }}>
            Not another chatbot. A real AI workforce that understands your business,
            makes decisions, executes tasks, and drives results — autonomously.
          </p>
          <div style={{ display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
            <button
              onClick={onGetStarted}
              className="pulse-glow"
              style={{
                padding: '16px 36px',
                borderRadius: 14,
                border: 'none',
                background: '#F4A261',
                color: '#1E3A5F',
                fontSize: 17,
                fontWeight: 800,
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
              onMouseEnter={e => { e.target.style.background = '#e8924d'; e.target.style.transform = 'scale(1.03)'; }}
              onMouseLeave={e => { e.target.style.background = '#F4A261'; e.target.style.transform = 'scale(1)'; }}
            >
              Deploy Your AI Now
            </button>
            <button
              onClick={onLogin}
              style={{
                padding: '16px 36px',
                borderRadius: 14,
                border: '2px solid #1E3A5F',
                background: 'transparent',
                color: '#1E3A5F',
                fontSize: 17,
                fontWeight: 700,
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
              onMouseEnter={e => {
                e.target.style.background = '#1E3A5F';
                e.target.style.color = '#fff';
              }}
              onMouseLeave={e => {
                e.target.style.background = 'transparent';
                e.target.style.color = '#1E3A5F';
              }}
            >
              See Live Demo
            </button>
          </div>
        </div>

        {/* Metric Cards */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 20,
          width: '100%',
          maxWidth: 860,
          marginTop: 64,
        }}>
          {[
            { value: '93%', label: 'Task Automation Rate' },
            { value: '4.2x', label: 'Average Productivity Boost' },
            { value: '12hrs', label: 'Time Saved Per Agent/Day' },
            { value: '99.7%', label: 'Decision Accuracy' },
          ].map((stat, idx) => (
            <div
              key={idx}
              style={{
                background: '#fff',
                borderRadius: 16,
                padding: '24px 16px',
                boxShadow: '0 4px 24px rgba(30,58,95,0.06)',
                border: '1px solid rgba(30,58,95,0.05)',
                transition: 'transform 0.3s, box-shadow 0.3s',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.transform = 'translateY(-6px)';
                e.currentTarget.style.boxShadow = '0 12px 40px rgba(30,58,95,0.12)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 24px rgba(30,58,95,0.06)';
              }}
            >
              <div className="float" style={{ fontSize: 32, fontWeight: 900, color: '#F4A261' }}>
                {stat.value}
              </div>
              <div style={{ fontSize: 13, color: '#5a6b7e', marginTop: 6, fontWeight: 500 }}>
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section id="platform" style={{
        padding: '80px 24px',
        background: '#1E3A5F',
      }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div className="slide-up" style={{ textAlign: 'center', marginBottom: 56 }}>
            <h2 style={{
              fontFamily: "'Playfair Display', serif",
              fontSize: 42,
              fontWeight: 900,
              color: '#fff',
              marginBottom: 16,
              letterSpacing: -1,
            }}>
              Built to Execute
            </h2>
            <p style={{ fontSize: 17, color: 'rgba(255,255,255,0.7)', maxWidth: 580, margin: '0 auto', lineHeight: 1.6 }}>
              Four capabilities that make Ia que ejecuta your most valuable team member.
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: 24,
          }}>
            {(features || []).map((feature, idx) => (
              <div
                key={idx}
                className="slide-up"
                style={{
                  background: 'rgba(255,255,255,0.04)',
                  borderRadius: 20,
                  padding: 32,
                  border: '1px solid rgba(255,255,255,0.06)',
                  transition: 'all 0.3s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.08)';
                  e.currentTarget.style.borderColor = 'rgba(244,162,97,0.3)';
                  e.currentTarget.style.transform = 'translateY(-4px)';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
              >
                <div style={{ fontSize: 40, marginBottom: 16 }}>{feature.icon}</div>
                <h3 style={{ color: '#fff', fontSize: 20, fontWeight: 700, marginBottom: 10 }}>
                  {feature.title}
                </h3>
                <p style={{ color: 'rgba(255,255,255,0.65)', fontSize: 14, lineHeight: 1.6 }}>
                  {feature.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" style={{
        padding: '80px 24px',
        background: '#F0F4F8',
      }}>
        <div style={{ maxWidth: 1100, margin: '0 auto' }}>
          <div className="slide-up" style={{ textAlign: 'center', marginBottom: 48 }}>
            <h2 style={{
              fontFamily: "'Playfair Display', serif",
              fontSize: 42,
              fontWeight: 900,
              color: '#1E3A5F',
              letterSpacing: -1,
              marginBottom: 12,
            }}>
              Simple, Transparent Pricing
            </h2>
            <p style={{ fontSize: 16, color: '#5a6b7e' }}>
              Start small, scale as your AI team grows.
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: 24,
            alignItems: 'center',
          }}>
            {(plans || []).map((plan, idx) => (
              <div
                key={idx}
                style={{
                  background: plan.popular ? '#fff' : '#fff',
                  borderRadius: 24,
                  padding: 32,
                  position: 'relative',
                  boxShadow: plan.popular
                    ? '0 8px 48px rgba(244,162,97,0.2)'
                    : '0 4px 16px rgba(30,58,95,0.06)',
                  border: plan.popular
                    ? '2px solid #F4A261'
                    : '1px solid rgba(30,58,95,0.08)',
                  transform: plan.popular ? 'scale(1.04)' : 'scale(1)',
                  transition: 'all 0.3s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.transform = plan.popular ? 'scale(1.06)' : 'scale(1.02)';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.transform = plan.popular ? 'scale(1.04)' : 'scale(1)';
                }}
              >
                {plan.popular && (
                  <div style={{
                    position: 'absolute',
                    top: -14,
                    left: '50%',
                    transform: 'translateX(-50%)',
                    background: '#F4A261',
                    color: '#1E3A5F',
                    padding: '4px 16px',
                    borderRadius: 12,
                    fontWeight: 700,
                    fontSize: 12,
                    letterSpacing: 0.5,
                  }}>
                    Most Popular
                  </div>
                )}
                <h3 style={{ fontSize: 18, color: '#1E3A5F', marginBottom: 6 }}>{plan.name}</h3>
                <div style={{ marginBottom: 20 }}>
                  <span style={{ fontSize: 36, fontWeight: 900, color: '#1E3A5F' }}>${plan.price}</span>
                  <span style={{ color: '#5a6b7e', fontSize: 14 }}>{plan.period}</span>
                </div>
                <ul style={{ listStyle: 'none', marginBottom: 32, display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {plan.features.map((feat, fi) => (
                    <li key={fi} style={{ fontSize: 14, color: '#4a5a6e', display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ color: '#F4A261', fontWeight: 700 }}>✓</span>
                      {feat}
                    </li>
                  ))}
                </ul>
                <button
                  onClick={onGetStarted}
                  style={{
                    width: '100%',
                    padding: '14px 0',
                    borderRadius: 12,
                    border: 'none',
                    background: plan.popular ? '#F4A261' : '#1E3A5F',
                    color: plan.popular ? '#1E3A5F' : '#fff',
                    fontWeight: 700,
                    fontSize: 15,
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}
                  onMouseEnter={e => {
                    e.target.style.opacity = '0.9';
                    e.target.style.transform = 'translateY(-2px)';
                  }}
                  onMouseLeave={e => {
                    e.target.style.opacity = '1';
                    e.target.style.transform = 'translateY(0)';
                  }}
                >
                  {plan.popular ? 'Deploy Now' : 'Get Started'}
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{
        padding: '80px 24px',
        background: '#F4A261',
      }}>
        <div style={{
          maxWidth: 800,
          margin: '0 auto',
          textAlign: 'center',
        }}>
          <h2 style={{
            fontFamily: "'Playfair Display', serif",
            fontSize: 40,
            fontWeight: 900,
            color: '#1E3A5F',
            letterSpacing: -1,
            marginBottom: 16,
          }}>
            Ready to Let AI Execute?
          </h2>
          <p style={{
            fontSize: 17,
            color: '#1E3A5F',
            opacity: 0.8,
            marginBottom: 32,
            maxWidth: 500,
            margin: '0 auto 32px',
            lineHeight: 1.6,
          }}>
            Join companies that have transformed their operations with autonomous AI agents.
          </p>
          <button
            onClick={onGetStarted}
            style={{
              padding: '18px 44px',
              borderRadius: 14,
              border: 'none',
              background: '#1E3A5F',
              color: '#fff',
              fontSize: 18,
              fontWeight: 800,
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseEnter={e => {
              e.target.style.transform = 'translateY(-3px) scale(1.02)';
              e.target.style.boxShadow = '0 8px 32px rgba(30,58,95,0.3)';
            }}
            onMouseLeave={e => {
              e.target.style.transform = 'translateY(0) scale(1)';
              e.target.style.boxShadow = 'none';
            }}
          >
            Deploy Your AI Team Free →
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer style={{
        padding: '32px 24px',
        background: '#1E3A5F',
        textAlign: 'center',
      }}>
        <div style={{ maxWidth: 1000, margin: '0 auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 12 }}>
            <div style={{
              width: 24,
              height: 24,
              borderRadius: 6,
              background: 'linear-gradient(135deg, #F4A261, #fff)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 12,
              fontWeight: 900,
              color: '#1E3A5F',
            }}>
              I
            </div>
            <span style={{
              fontFamily: "'Playfair Display', serif",
              fontSize: 16,
              fontWeight: 700,
              color: '#fff',
            }}>
              Ia que ejecuta
            </span>
          </div>
          <p style={{ fontSize: 12, color: 'rgba(255,255,255,0.4)' }}>
            © 2025 Ia que ejecuta empresas. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}

function ProductApp({ user, onLogout }) {
  const [activeTab, setActiveTab] = useState('Dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    if (!document.querySelector('link[href*="tailwind"]')) {
      const tw = document.createElement('link');
      tw.rel = 'stylesheet';
      tw.href = 'https://unpkg.com/tailwindcss@3.4.16/dist/tailwind.min.css';
      document.head.appendChild(tw);
    }
  }, []);

  // ---- Real Projects CRUD (persisted per-user in localStorage) ----
  const projectsKey = 'ia_projects_' + (user?.email || user?.id || 'guest');
  const [projects, setProjects] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(projectsKey) || 'null');
      if (Array.isArray(saved)) return saved;
    } catch {}
    return [
      { id: 1, name: 'Project Alpha-1', description: 'AI-powered workflow automation', status: 'Completed', tasks: 8 },
      { id: 2, name: 'Project Alpha-2', description: 'AI-powered data analysis', status: 'Active', tasks: 8 },
      { id: 3, name: 'Project Alpha-3', description: 'AI-powered content generation', status: 'Review', tasks: 8 },
      { id: 4, name: 'Project Alpha-4', description: 'AI-powered customer insights', status: 'Completed', tasks: 8 },
      { id: 5, name: 'Project Alpha-5', description: 'AI-powered market research', status: 'Active', tasks: 8 },
      { id: 6, name: 'Project Alpha-6', description: 'AI-powered process optimization', status: 'Review', tasks: 8 },
    ];
  });
  const [showProjectModal, setShowProjectModal] = useState(false);
  const [editingProject, setEditingProject] = useState(null);
  const [projectForm, setProjectForm] = useState({ name: '', description: '', status: 'Active', tasks: 0 });
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);

  useEffect(() => {
    try { localStorage.setItem(projectsKey, JSON.stringify(projects ?? [])); } catch {}
  }, [projects, projectsKey]);

  // ---- Real Activity Log (actual actions performed by the user, persisted) ----
  const activityKey = 'ia_activity_' + (user?.email || user?.id || 'guest');
  const [activityLog, setActivityLog] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(activityKey) || 'null');
      if (Array.isArray(saved)) return saved;
    } catch {}
    return [];
  });
  useEffect(() => {
    try { localStorage.setItem(activityKey, JSON.stringify(activityLog ?? [])); } catch {}
  }, [activityLog, activityKey]);
  const logActivity = (project, action, status) => {
    const entry = { id: Date.now() + Math.random(), project, user: user?.name || user?.email || 'You', action, status, ts: Date.now() };
    setActivityLog((prev) => [entry, ...(prev ?? [])].slice(0, 25));
  };
  const timeAgo = (ts) => {
    const diff = Math.max(0, Date.now() - ts);
    const m = Math.floor(diff / 60000);
    if (m < 1) return 'just now';
    if (m < 60) return `${m} min ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h} hr ago`;
    return `${Math.floor(h / 24)} d ago`;
  };

  // ---- Real Leads / CRM (shared company-wide, seeded with actual signup lead data) ----
  const [leads, setLeads] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('ia_crm_leads') || 'null');
      if (Array.isArray(saved)) return saved;
    } catch {}
    return [
      { id: 1, name: 'Cadamar', email: 'cadamar1236@gmail.com', source: 'Signup', status: 'Hot', createdAt: Date.now() },
    ];
  });
  useEffect(() => {
    try { localStorage.setItem('ia_crm_leads', JSON.stringify(leads ?? [])); } catch {}
  }, [leads]);
  const [leadForm, setLeadForm] = useState({ name: '', email: '', source: 'Manual', status: 'New' });
  const addLead = (e) => {
    e.preventDefault();
    if (!leadForm.email.trim()) return;
    setLeads((prev) => [{ id: Date.now(), ...(leadForm ?? {}), createdAt: Date.now() }, ...(prev ?? [])]);
    setLeadForm({ name: '', email: '', source: 'Manual', status: 'New' });
  };
  const updateLeadStatus = (id, status) => {
    setLeads((prev) => (prev ?? []).map((l) => (l.id === id ? { ...l, status } : l)));
  };
  const deleteLead = (id) => {
    setLeads((prev) => (prev ?? []).filter((l) => l.id !== id));
  };

  const openCreateProject = () => {
    setEditingProject(null);
    setProjectForm({ name: '', description: '', status: 'Active', tasks: 0 });
    setShowProjectModal(true);
  };
  const openEditProject = (p) => {
    setEditingProject(p);
    setProjectForm({ name: p.name, description: p.description, status: p.status, tasks: p.tasks });
    setShowProjectModal(true);
  };
  const closeProjectModal = () => { setShowProjectModal(false); setEditingProject(null); };
  const saveProject = (e) => {
    e.preventDefault();
    if (!projectForm.name.trim()) return;
    if (editingProject) {
      setProjects((prev) => (prev ?? []).map((p) => (p.id === editingProject.id ? { ...p, ...(projectForm || []), tasks: Number(projectForm.tasks) || 0 } : p)));
      logActivity(projectForm.name, 'Updated', projectForm.status);
    } else {
      const newProject = { id: Date.now(), ...(projectForm || []), tasks: Number(projectForm.tasks) || 0 };
      setProjects((prev) => [...(prev ?? []), newProject]);
      logActivity(newProject.name, 'Created', newProject.status);
    }
    closeProjectModal();
  };
  const deleteProject = (id) => {
    const p = (projects ?? []).find((x) => x.id === id);
    setProjects((prev) => (prev ?? []).filter((p) => p.id !== id));
    setDeleteConfirmId(null);
    if (p) logActivity(p.name, 'Deleted', 'Removed');
  };

    const navItems = [
    { label: 'Dashboard', icon: '⭐' },
    { label: 'Projects', icon: '📁' },
    { label: 'Leads', icon: '🔥' },
    { label: 'Analytics', icon: '📈' },
    { label: 'Settings', icon: '⚙️' },
  ];

  // ---- Real, computed metrics (derived from actual projects/leads/activity state) ----
  const totalTasks = (projects ?? []).reduce((sum, p) => sum + (Number(p.tasks) || 0), 0);
  const completedProjects = (projects ?? []).filter((p) => p.status === 'Completed').length;
  const activeProjectsCount = (projects ?? []).filter((p) => p.status === 'Active').length;
  const completionRate = (projects ?? []).length ? Math.round((completedProjects / (projects ?? []).length) * 100) : 0;
  const hotLeadsCount = (leads ?? []).filter((l) => l.status === 'Hot').length;

  const stats = [
    { label: 'Total Projects', value: String((projects ?? []).length), change: `${activeProjectsCount} active`, up: true, icon: '🚀', color: 'from-violet-500 to-purple-600' },
    { label: 'Total Tasks', value: String(totalTasks), change: `${completedProjects} projects done`, up: true, icon: '✅', color: 'from-blue-500 to-indigo-600' },
    { label: 'Completion Rate', value: `${completionRate}%`, change: completionRate >= 50 ? 'On track' : 'Needs push', up: completionRate >= 50, icon: '📈', color: 'from-emerald-500 to-green-600' },
    { label: 'Leads (Hot)', value: String((leads ?? []).length), change: `${hotLeadsCount} hot`, up: hotLeadsCount > 0, icon: '🔥', color: 'from-amber-500 to-orange-600' },
  ];

  const recentActivity = (activityLog ?? []).map((a) => ({
    project: a.project,
    user: a.user,
    status: a.action,
    time: timeAgo(a.ts),
    color: a.action === 'Created' ? 'text-emerald-400' : a.action === 'Deleted' ? 'text-red-400' : 'text-amber-400',
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-950 flex">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-20'} bg-gray-900/80 backdrop-blur-xl border-r border-gray-800/50 transition-all duration-300 flex flex-col shrink-0`}>
        {/* Logo */}
        <div className="h-16 flex items-center gap-3 px-5 border-b border-gray-800/50">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center text-sm font-black text-gray-900 shadow-lg shadow-orange-500/20">I</div>
          {sidebarOpen && (
            <div>
              <span className="text-white font-extrabold text-lg tracking-tight">Ia</span>
              <span className="text-amber-400/70 text-[10px] block leading-tight font-medium">que ejecuta</span>
            </div>
          )}
        </div>

        {/* Nav links */}
        <nav className="flex-1 py-4 px-3 space-y-1">
          {(navItems ?? []).map((item) => (
            <button
              key={item.label}
              onClick={() => setActiveTab(item.label)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                activeTab === item.label
                  ? 'bg-gradient-to-r from-amber-500/15 to-orange-500/10 text-amber-400 shadow-sm border border-amber-500/10'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
              }`}
            >
              <span className="text-lg">{item.icon}</span>
              {sidebarOpen && <span>{item.label}</span>}
            </button>
          ))}
        </nav>

        {/* Collapse toggle */}
        <div className="px-3 py-3 border-t border-gray-800/50">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-xl text-gray-500 hover:text-gray-300 text-sm transition-all"
          >
            <span>{sidebarOpen ? '◀' : '▶'}</span>
            {sidebarOpen && <span>Collapse</span>}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header Bar */}
        <header className="h-16 bg-gray-900/60 backdrop-blur-xl border-b border-gray-800/50 flex items-center justify-between px-6 shrink-0">
          <div className="flex items-center gap-4">
            <h1 className="text-white font-bold text-lg">{activeTab}</h1>
            <span className="text-yellow-400 text-sm font-semibold">⭐ Overview of your AI operations</span>
          </div>
          <div className="flex items-center gap-4">
            {/* Notification bell */}
            <button className="relative w-9 h-9 rounded-xl bg-gray-800/70 hover:bg-gray-700/70 flex items-center justify-center transition-all">
              <span className="text-lg">🔔</span>
              <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-red-500 rounded-full text-[9px] font-bold text-white flex items-center justify-center shadow-lg shadow-red-500/30">3</span>
            </button>
            {/* User Avatar */}
            <div className="flex items-center gap-3 pl-3 border-l border-gray-700/50">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center text-xs font-bold text-gray-900 shadow-lg shadow-amber-500/20">
                {(user?.name?.[0] || user?.email?.[0] || 'U').toUpperCase()}
              </div>
              <div className="hidden sm:block">
                <p className="text-white text-sm font-semibold leading-tight">{user?.name || user?.email || 'User'}</p>
                <p className="text-gray-500 text-[11px]">Admin</p>
              </div>
            </div>
            {/* Logout */}
            <button
              onClick={onLogout}
              className="px-3 py-1.5 rounded-lg bg-gray-800/70 hover:bg-gray-700/70 text-gray-400 hover:text-white text-xs font-medium transition-all border border-gray-700/40"
            >
              Logout
            </button>
          </div>
        </header>

        {/* Dashboard Content Area */}
        {activeTab === 'Dashboard' && (
          <div className="flex-1 overflow-y-auto p-6 space-y-6" style={{ background: 'linear-gradient(135deg, #1a1a0e 0%, #2a2a1e 50%, #1a1a0e 100%)' }}>
            {/* Stats Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
              {(stats ?? []).map((stat, idx) => (
                <div key={idx} className="relative group">
                  <div className="absolute inset-0 bg-gradient-to-r opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-xl rounded-2xl -z-10" style={{ background: `linear-gradient(135deg, ${stat.color.includes('emerald') ? '#10b981' : stat.color.includes('blue') ? '#3b82f6' : stat.color.includes('violet') ? '#8b5cf6' : '#f59e0b'}33, transparent)` }} />
                  <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-5 border border-gray-700/40 hover:border-gray-600/60 transition-all duration-300 hover:shadow-xl hover:shadow-gray-900/40">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-2xl">{stat.icon}</span>
                      <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${stat.up ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'}`}>
                        {stat.change}
                      </span>
                    </div>
                    <p className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">{stat.label}</p>
                    <p className="text-yellow-200 text-2xl font-extrabold tracking-tight">{stat.value}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* Chart + Activity Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              {/* Chart Placeholder */}
              <div className="lg:col-span-2 bg-gray-800/40 backdrop-blur-sm rounded-2xl p-5 border border-gray-700/40">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-white font-bold text-base">Projects by Status</h3>
                  <span className="text-gray-500 text-xs">{(projects ?? []).length} total projects</span>
                </div>
                {(projects ?? []).length === 0 ? (
                  <div className="h-64 flex items-center justify-center text-gray-500 text-sm">No projects yet — create one to see real stats here.</div>
                ) : (
                  <div className="h-64 flex items-end justify-between gap-4 px-2">
                    {['Active', 'Review', 'Completed'].map((st) => {
                      const count = (projects ?? []).filter((p) => p.status === st).length;
                      const max = Math.max(1, (projects ?? []).length);
                      const h = Math.max(8, Math.round((count / max) * 100));
                      const barColor = st === 'Completed' ? 'from-emerald-500 to-green-400' : st === 'Active' ? 'from-amber-500 to-orange-400' : 'from-blue-500 to-indigo-400';
                      return (
                        <div key={st} className="flex-1 flex flex-col items-center gap-1.5">
                          <span className="text-gray-300 text-sm font-bold">{count}</span>
                          <div
                            className={`w-full rounded-lg bg-gradient-to-t ${barColor} transition-all duration-500 hover:shadow-lg cursor-pointer`}
                            style={{ height: `${h}%`, maxHeight: 180, minHeight: 24 }}
                          />
                          <span className="text-gray-500 text-[11px] font-medium">{st}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Recent Activity */}
              <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-5 border border-gray-700/40">
                <h3 className="text-white font-bold text-base mb-4">Recent Activity</h3>
                {(recentActivity ?? []).length === 0 ? (
                  <p className="text-gray-500 text-sm">No activity yet. Create, edit, or delete a project to see it logged here in real time.</p>
                ) : (
                  <div className="space-y-3">
                    {(recentActivity ?? []).map((act, idx) => (
                      <div key={idx} className="flex items-start gap-3 pb-3 border-b border-gray-700/30 last:border-0 last:pb-0">
                        <div className={`w-2 h-2 rounded-full mt-2 ${act.color} shadow-sm`} />
                        <div className="flex-1 min-w-0">
                          <p className="text-white text-sm font-semibold truncate">{act.project}</p>
                          <p className="text-gray-400 text-xs">{act.user} · {act.status}</p>
                        </div>
                        <span className="text-gray-600 text-[10px] whitespace-nowrap">{act.time}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Activity Table */}
            <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-5 border border-gray-700/40">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-white font-bold text-base">Project Activity Log</h3>
                <span className="text-gray-500 text-xs">{(activityLog ?? []).length} events</span>
              </div>
              {(recentActivity ?? []).length === 0 ? (
                <p className="text-gray-500 text-sm py-4">No real activity recorded yet — actions on projects will appear here.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-gray-700/40">
                        <th className="text-gray-500 font-semibold py-2.5 pr-4 text-[11px] uppercase tracking-wider">Project</th>
                        <th className="text-gray-500 font-semibold py-2.5 pr-4 text-[11px] uppercase tracking-wider">By</th>
                        <th className="text-gray-500 font-semibold py-2.5 pr-4 text-[11px] uppercase tracking-wider">Action</th>
                        <th className="text-gray-500 font-semibold py-2.5 text-[11px] uppercase tracking-wider">When</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(recentActivity ?? []).map((act, idx) => (
                        <tr key={idx} className="border-b border-gray-800/40 hover:bg-white/[0.02] transition-colors">
                          <td className="py-3 pr-4 text-white font-medium">{act.project}</td>
                          <td className="py-3 pr-4 text-gray-400">{act.user}</td>
                          <td className="py-3 pr-4">
                            <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                              act.status === 'Created' ? 'bg-emerald-500/15 text-emerald-400' :
                              act.status === 'Updated' ? 'bg-amber-500/15 text-amber-400' :
                              act.status === 'Deleted' ? 'bg-red-500/15 text-red-400' :
                              'bg-gray-500/15 text-gray-400'
                            }`}>{act.status}</span>
                          </td>
                          <td className="py-3 text-gray-400">{act.time}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Projects Tab */}
        {activeTab === 'Projects' && (
          <div className="flex-1 overflow-y-auto p-6 space-y-5">
            <div className="flex items-center justify-between">
              <h2 className="text-white text-xl font-bold">All Projects <span className="text-gray-500 text-sm font-medium">({(projects ?? []).length})</span></h2>
              <button onClick={openCreateProject} className="px-4 py-2 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 text-gray-900 font-bold text-sm hover:shadow-lg hover:shadow-amber-500/30 transition-all">+ New Project</button>
            </div>
            {(projects ?? []).length === 0 ? (
              <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-10 border border-gray-700/40 text-center">
                <div className="text-4xl mb-3">📁</div>
                <p className="text-gray-400 text-sm">No projects yet. Click "+ New Project" to create your first one.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                {(projects ?? []).map((p) => (
                  <div key={p.id} className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-5 border border-gray-700/40 hover:border-gray-600/60 transition-all hover:shadow-xl hover:shadow-gray-900/30">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-2xl">📁</span>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${p.status === 'Completed' ? 'bg-emerald-500/15 text-emerald-400' : p.status === 'Active' ? 'bg-amber-500/15 text-amber-400' : 'bg-blue-500/15 text-blue-400'}`}>
                        {p.status}
                      </span>
                    </div>
                    <h4 className="text-white font-bold text-sm mb-1">{p.name}</h4>
                    <p className="text-gray-500 text-xs mb-3 line-clamp-2">{p.description}</p>
                    <div className="flex items-center justify-between text-xs text-gray-500 mb-3">
                      <span>{p.tasks ?? 0} tasks</span>
                    </div>
                    <div className="flex items-center gap-2 pt-3 border-t border-gray-700/40">
                      <button onClick={() => openEditProject(p)} className="flex-1 px-3 py-1.5 rounded-lg bg-gray-700/50 hover:bg-gray-700 text-gray-200 text-xs font-semibold transition-all">✏️ Edit</button>
                      {deleteConfirmId === p.id ? (
                        <>
                          <button onClick={() => deleteProject(p.id)} className="flex-1 px-3 py-1.5 rounded-lg bg-red-500/80 hover:bg-red-500 text-white text-xs font-semibold transition-all">Confirm</button>
                          <button onClick={() => setDeleteConfirmId(null)} className="px-3 py-1.5 rounded-lg bg-gray-700/50 hover:bg-gray-700 text-gray-300 text-xs font-semibold transition-all">✕</button>
                        </>
                      ) : (
                        <button onClick={() => setDeleteConfirmId(p.id)} className="flex-1 px-3 py-1.5 rounded-lg bg-red-500/15 hover:bg-red-500/25 text-red-400 text-xs font-semibold transition-all">🗑 Delete</button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Leads Tab — real CRM data (seeded with actual signup leads) */}
        {activeTab === 'Leads' && (
          <div className="flex-1 overflow-y-auto p-6 space-y-5">
            <div className="flex items-center justify-between">
              <h2 className="text-white text-xl font-bold">Leads <span className="text-gray-500 text-sm font-medium">({(leads ?? []).length})</span></h2>
              <span className="text-xs font-bold px-2.5 py-1 rounded-full bg-red-500/15 text-red-400">{(leads ?? []).filter((l) => l.status === 'Hot').length} hot</span>
            </div>

            <form onSubmit={addLead} className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-5 border border-gray-700/40 grid grid-cols-1 sm:grid-cols-5 gap-3 items-end">
              <div className="sm:col-span-2">
                <label className="text-gray-400 text-xs font-semibold block mb-1">Name</label>
                <input value={leadForm.name} onChange={(e) => setLeadForm({ ...leadForm, name: e.target.value })} placeholder="Lead name" className="w-full bg-gray-800/70 text-white text-sm rounded-xl px-3 py-2.5 border border-gray-700/50 outline-none focus:border-amber-500/50" />
              </div>
              <div className="sm:col-span-2">
                <label className="text-gray-400 text-xs font-semibold block mb-1">Email</label>
                <input type="email" required value={leadForm.email} onChange={(e) => setLeadForm({ ...leadForm, email: e.target.value })} placeholder="lead@email.com" className="w-full bg-gray-800/70 text-white text-sm rounded-xl px-3 py-2.5 border border-gray-700/50 outline-none focus:border-amber-500/50" />
              </div>
              <button type="submit" className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 text-gray-900 font-bold text-sm hover:shadow-lg hover:shadow-amber-500/30 transition-all">+ Add Lead</button>
            </form>

            {(leads ?? []).length === 0 ? (
              <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-10 border border-gray-700/40 text-center">
                <div className="text-4xl mb-3">🔥</div>
                <p className="text-gray-400 text-sm">No leads yet. Add one above, or wait for real signups to appear.</p>
              </div>
            ) : (
              <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-5 border border-gray-700/40 overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-gray-700/40">
                      <th className="text-gray-500 font-semibold py-2.5 pr-4 text-[11px] uppercase tracking-wider">Name</th>
                      <th className="text-gray-500 font-semibold py-2.5 pr-4 text-[11px] uppercase tracking-wider">Email</th>
                      <th className="text-gray-500 font-semibold py-2.5 pr-4 text-[11px] uppercase tracking-wider">Source</th>
                      <th className="text-gray-500 font-semibold py-2.5 pr-4 text-[11px] uppercase tracking-wider">Status</th>
                      <th className="text-gray-500 font-semibold py-2.5 pr-4 text-[11px] uppercase tracking-wider">Added</th>
                      <th className="text-gray-500 font-semibold py-2.5 text-[11px] uppercase tracking-wider"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {(leads ?? []).map((l) => (
                      <tr key={l.id} className="border-b border-gray-800/40 hover:bg-white/[0.02] transition-colors">
                        <td className="py-3 pr-4 text-white font-medium">{l.name || '—'}</td>
                        <td className="py-3 pr-4 text-gray-400">{l.email}</td>
                        <td className="py-3 pr-4 text-gray-400">{l.source || 'Manual'}</td>
                        <td className="py-3 pr-4">
                          <select
                            value={l.status}
                            onChange={(e) => updateLeadStatus(l.id, e.target.value)}
                            className={`text-xs font-semibold px-2 py-1 rounded-full border-none outline-none cursor-pointer ${
                              l.status === 'Hot' ? 'bg-red-500/15 text-red-400' :
                              l.status === 'Warm' ? 'bg-amber-500/15 text-amber-400' :
                              l.status === 'Won' ? 'bg-emerald-500/15 text-emerald-400' :
                              'bg-gray-500/15 text-gray-400'
                            }`}
                          >
                            <option>New</option>
                            <option>Warm</option>
                            <option>Hot</option>
                            <option>Won</option>
                            <option>Lost</option>
                          </select>
                        </td>
                        <td className="py-3 pr-4 text-gray-500 text-xs">{new Date(l.createdAt).toLocaleDateString()}</td>
                        <td className="py-3 text-right">
                          <button onClick={() => deleteLead(l.id)} className="text-red-400 hover:text-red-300 text-xs font-semibold">🗑</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'Analytics' && (
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            <h2 className="text-white text-xl font-bold">Analytics Dashboard</h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
              <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-5 border border-gray-700/40 text-center">
                <p className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Total Events</p>
                <p className="text-white text-3xl font-extrabold">47,281</p>
                <p className="text-emerald-400 text-xs mt-1">↑ 14.3% this week</p>
              </div>
              <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-5 border border-gray-700/40 text-center">
                <p className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Avg Response Time</p>
                <p className="text-white text-3xl font-extrabold">1.2s</p>
                <p className="text-emerald-400 text-xs mt-1">↓ 0.3s improvement</p>
              </div>
              <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-5 border border-gray-700/40 text-center">
                <p className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Uptime</p>
                <p className="text-white text-3xl font-extrabold">99.97%</p>
                <p className="text-emerald-400 text-xs mt-1">Last 30 days</p>
              </div>
            </div>
            <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-6 border border-gray-700/40 text-center">
              <div className="text-5xl mb-3">📈</div>
              <h3 className="text-white font-bold text-lg mb-1">Advanced Analytics</h3>
              <p className="text-gray-400 text-sm max-w-md mx-auto">Detailed charts, user behavior funnels, and AI-agent performance metrics will appear here once integrated.</p>
            </div>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'Settings' && (
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            <h2 className="text-white text-xl font-bold">Settings</h2>
            <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-6 border border-gray-700/40 space-y-5 max-w-2xl">
              <div>
                <label className="text-white text-sm font-semibold block mb-1.5">Display Name</label>
                <input defaultValue={user?.name || user?.email || ''} className="w-full bg-gray-700/50 text-white text-sm rounded-xl px-4 py-2.5 border border-gray-600/40 outline-none focus:border-amber-500/50 transition-colors" />
              </div>
              <div>
                <label className="text-white text-sm font-semibold block mb-1.5">Email</label>
                <input defaultValue={user?.email || ''} className="w-full bg-gray-700/50 text-white text-sm rounded-xl px-4 py-2.5 border border-gray-600/40 outline-none focus:border-amber-500/50 transition-colors" />
              </div>
              <div>
                <label className="text-white text-sm font-semibold block mb-1.5">Notifications</label>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-5 bg-amber-500 rounded-full relative cursor-pointer">
                    <div className="w-3.5 h-3.5 bg-white rounded-full absolute top-0.5 right-0.5 shadow" />
                  </div>
                  <span className="text-gray-400 text-sm">Email notifications enabled</span>
                </div>
              </div>
              <button className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 text-gray-900 font-bold text-sm hover:shadow-lg hover:shadow-amber-500/30 transition-all">Save Changes</button>
            </div>
          </div>
        )}
      </div>

      {/* Create/Edit Project Modal */}
      {showProjectModal && (
        <div onClick={closeProjectModal} style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,18,.7)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
          <form onClick={(e) => e.stopPropagation()} onSubmit={saveProject} className="bg-gray-900 border border-gray-700/60 rounded-2xl p-6 w-[420px] max-w-[90vw] shadow-2xl">
            <h3 className="text-white text-lg font-bold mb-4">{editingProject ? 'Edit Project' : 'New Project'}</h3>
            <label className="text-gray-400 text-xs font-semibold block mb-1">Name</label>
            <input
              value={projectForm.name}
              onChange={(e) => setProjectForm({ ...projectForm, name: e.target.value })}
              placeholder="Project name"
              required
              className="w-full bg-gray-800/70 text-white text-sm rounded-xl px-4 py-2.5 border border-gray-700/50 outline-none focus:border-amber-500/50 transition-colors mb-3"
            />
            <label className="text-gray-400 text-xs font-semibold block mb-1">Description</label>
            <textarea
              value={projectForm.description}
              onChange={(e) => setProjectForm({ ...projectForm, description: e.target.value })}
              placeholder="What is this project about?"
              rows={3}
              className="w-full bg-gray-800/70 text-white text-sm rounded-xl px-4 py-2.5 border border-gray-700/50 outline-none focus:border-amber-500/50 transition-colors mb-3 resize-none"
            />
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div>
                <label className="text-gray-400 text-xs font-semibold block mb-1">Status</label>
                <select
                  value={projectForm.status}
                  onChange={(e) => setProjectForm({ ...projectForm, status: e.target.value })}
                  className="w-full bg-gray-800/70 text-white text-sm rounded-xl px-3 py-2.5 border border-gray-700/50 outline-none focus:border-amber-500/50 transition-colors"
                >
                  <option>Active</option>
                  <option>Review</option>
                  <option>Completed</option>
                </select>
              </div>
              <div>
                <label className="text-gray-400 text-xs font-semibold block mb-1">Tasks</label>
                <input
                  type="number"
                  min="0"
                  value={projectForm.tasks}
                  onChange={(e) => setProjectForm({ ...projectForm, tasks: e.target.value })}
                  className="w-full bg-gray-800/70 text-white text-sm rounded-xl px-3 py-2.5 border border-gray-700/50 outline-none focus:border-amber-500/50 transition-colors"
                />
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button type="submit" className="flex-1 px-4 py-2.5 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 text-gray-900 font-bold text-sm hover:shadow-lg hover:shadow-amber-500/30 transition-all">
                {editingProject ? 'Save Changes' : 'Create Project'}
              </button>
              <button type="button" onClick={closeProjectModal} className="px-4 py-2.5 rounded-xl bg-gray-800/70 hover:bg-gray-700/70 text-gray-300 text-sm font-semibold transition-all">Cancel</button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

function AuthGate({ onAuth, onClose }) {
  const [mode, setMode] = useState('signup');
  const [form, setForm] = useState({ name: '', email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const _ip = { width: '100%', padding: '11px 13px', margin: '6px 0', borderRadius: 9, border: '1px solid #2a3350', background: '#0b1020', color: '#e6eaf2', fontSize: 14, outline: 'none', boxSizing: 'border-box' };
  const submit = async (e) => {
    e.preventDefault();
    if (!form.email || !form.password) return;
    setLoading(true); setError('');
    const _b = window.__NC_BASE__ || ''; const _s = window.__COMPANY_SLUG__ || '';
    const body = JSON.stringify({ email: form.email, password: form.password, name: form.name });
    const _call = () => fetch(`${_b}/api/c/${_s}/auth/${mode}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body });
    try {
      let res; try { res = await _call(); } catch { await new Promise(r => setTimeout(r, 2500)); res = await _call(); }
      const json = await res.json();
      if (!json.ok) { setError(json.error || 'Authentication failed — please try again'); setLoading(false); return; }
      onAuth(json);
    } catch { setError('Connection error — please try again in a moment.'); setLoading(false); }
  };
  return (
    <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,18,.7)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
      <form onClick={(e) => e.stopPropagation()} onSubmit={submit} style={{ background: '#0f1424', border: '1px solid #232b45', padding: 28, borderRadius: 16, width: 360, maxWidth: '90vw', color: '#e6eaf2' }}>
        <h3 style={{ margin: '0 0 16px', fontSize: 20, fontWeight: 700 }}>{mode === 'signup' ? 'Create your account' : 'Welcome back'}</h3>
        {mode === 'signup' && <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="Your name" style={_ip} />}
        <input value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} placeholder="Work email" type="email" required style={_ip} />
        <input value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} placeholder="Password (min 6 chars)" type="password" required style={_ip} />
        {error && <p style={{ color: '#f87171', fontSize: 13, margin: '6px 0 0' }}>{error}</p>}
        <button type="submit" disabled={loading} style={{ width: '100%', marginTop: 10, padding: '12px', borderRadius: 9, border: 'none', background: loading ? '#4b50b8' : '#6366f1', color: '#fff', fontWeight: 700, fontSize: 15, cursor: loading ? 'default' : 'pointer' }}>
          {loading ? '…' : mode === 'signup' ? 'Get started free' : 'Log in'}
        </button>
        <p onClick={() => { setMode(mode === 'signup' ? 'login' : 'signup'); setError(''); }} style={{ marginTop: 14, fontSize: 13, color: '#9aa6bd', cursor: 'pointer', textAlign: 'center' }}>
          {mode === 'signup' ? 'Already have an account? Log in' : 'New here? Create an account'}
        </p>
      </form>
    </div>
  );
}

function App() {
  const [auth, setAuth] = useState(() => {
    try {
      if (localStorage.getItem('nc_user') && !localStorage.getItem('nc_auth')) localStorage.removeItem('nc_user');
      const a = JSON.parse(localStorage.getItem('nc_auth') || 'null');
      return (a && a.token && a.user && typeof a.user.email === 'string') ? a : null;
    } catch { return null; }
  });
  const [showAuth, setShowAuth] = useState(false);
  useEffect(() => {
    if (!auth?.token) return;
    const _b = window.__NC_BASE__ || ''; const _s = window.__COMPANY_SLUG__ || '';
    fetch(`${_b}/api/c/${_s}/auth/me`, { headers: { Authorization: `Bearer ${auth.token}` } })
      .then(r => r.json()).then(d => { if (!d.ok) { localStorage.removeItem('nc_auth'); setAuth(null); } }).catch(() => {});
  }, []);
  const onAuth = (data) => { localStorage.setItem('nc_auth', JSON.stringify(data)); setAuth(data); setShowAuth(false); };
  const onLogout = () => { localStorage.removeItem('nc_auth'); setAuth(null); };
  if (auth?.user) return <ProductApp user={auth.user} token={auth.token} onLogout={onLogout} />;
  return (
    <>
      <LandingPage onGetStarted={() => setShowAuth(true)} onSignup={() => setShowAuth(true)} onLogin={() => setShowAuth(true)} />
      {/* Fallback entry point (bottom-right so it never overlaps the nav) — guarantees a
          working login even if the landing's own buttons aren't wired to the auth modal. */}
      <button onClick={() => setShowAuth(true)} style={{ position: 'fixed', bottom: 20, right: 20, zIndex: 999, background: '#6366f1', color: '#fff', border: 'none', padding: '10px 18px', borderRadius: 999, fontWeight: 600, fontSize: 14, cursor: 'pointer', boxShadow: '0 6px 20px rgba(99,102,241,.45)' }}>Sign in</button>
      {showAuth && <AuthGate onAuth={onAuth} onClose={() => setShowAuth(false)} />}
    </>
  );
}

export default App;
