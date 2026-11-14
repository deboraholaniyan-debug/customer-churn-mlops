
"use client";

import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Bell,
  ChevronDown,
  CircleHelp,
  Clock3,
  CreditCard,
  FileUp,
  LayoutDashboard,
  Menu,
  Search,
  Settings,
  ShieldCheck,
  Users,
  X,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const trendData = [
  { month: "May", customers: 390, churned: 74 },
  { month: "Jun", customers: 420, churned: 81 },
  { month: "Jul", customers: 405, churned: 77 },
  { month: "Aug", customers: 460, churned: 92 },
  { month: "Sep", customers: 445, churned: 85 },
  { month: "Oct", customers: 490, churned: 103 },
  { month: "Nov", customers: 470, churned: 96 },
];

const segmentData = [
  { name: "Low risk", value: 52, color: "#14b8a6" },
  { name: "Medium risk", value: 29, color: "#f59e0b" },
  { name: "High risk", value: 19, color: "#f87171" },
];

const recentPredictions = [
  { id: "CUS-1048", contract: "Month-to-month", probability: 0.91, risk: "High", date: "Nov 14, 2026" },
  { id: "CUS-1047", contract: "One year", probability: 0.67, risk: "Medium", date: "Nov 14, 2026" },
  { id: "CUS-1046", contract: "Two year", probability: 0.18, risk: "Low", date: "Nov 13, 2026" },
  { id: "CUS-1045", contract: "Month-to-month", probability: 0.84, risk: "High", date: "Nov 13, 2026" },
];

const navigation = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "Customers", icon: Users },
  { label: "Upload data", icon: FileUp },
  { label: "Predictions", icon: Zap },
  { label: "Insights", icon: BarChart3 },
  { label: "History", icon: Clock3 },
];

export default function Home() {
  const [activePage, setActivePage] = useState("Overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <main className="dashboard-shell">
      {sidebarOpen && (
        <button
          className="sidebar-backdrop"
          aria-label="Close navigation"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Activity size={22} /></div>
          <div>
            <div className="brand-name">ChurnOps</div>
            <div className="brand-caption">INTELLIGENCE PLATFORM</div>
          </div>
          <button className="mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Close menu">
            <X size={19} />
          </button>
        </div>

        <div className="workspace-label">WORKSPACE</div>
        <button className="workspace-switch">
          <div className="workspace-avatar">AC</div>
          <span className="workspace-text">
            <strong>Acme Corporation</strong>
            <small>Business workspace</small>
          </span>
          <ChevronDown size={16} />
        </button>

        <div className="nav-label">MAIN MENU</div>
        <nav className="navigation">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              onClick={() => {
                setActivePage(label);
                setSidebarOpen(false);
              }}
              className={`nav-item ${activePage === label ? "nav-active" : ""}`}
            >
              <Icon size={18} />
              <span>{label}</span>
              {label === "Predictions" && <span className="nav-count">2</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <div className="status-icon"><ShieldCheck size={18} /></div>
            <div>
              <strong>Model service</strong>
              <span><i className="status-dot" /> API connection pending</span>
            </div>
          </div>
          <button className="nav-item">
            <Settings size={18} /><span>Settings</span>
          </button>
          <button className="nav-item">
            <CircleHelp size={18} /><span>Help & support</span>
          </button>
          <div className="profile">
            <div className="profile-avatar">DO</div>
            <div className="profile-info">
              <strong>Workspace Admin</strong>
              <span>Administrator</span>
            </div>
            <ChevronDown size={16} />
          </div>
        </div>
      </aside>

      <section className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
              <Menu size={21} />
            </button>
            <div className="breadcrumb">Workspace <span>/</span> <strong>{activePage}</strong></div>
          </div>
          <div className="topbar-actions">
            <div className="environment-pill"><span /> Demo environment</div>
            <button className="icon-button" aria-label="Search"><Search size={19} /></button>
            <button className="icon-button notification-button" aria-label="Notifications">
              <Bell size={19} /><i />
            </button>
            <div className="topbar-divider" />
            <div className="topbar-date">14 November 2026</div>
          </div>
        </header>

        <div className="content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">CUSTOMER INTELLIGENCE</div>
              <h1>{activePage === "Overview" ? "Dashboard overview" : activePage}</h1>
              <p>
                {activePage === "Overview"
                  ? "Monitor customer retention, understand churn patterns and identify at-risk customers."
                  : `Explore your ${activePage.toLowerCase()} workspace.`}
              </p>
            </div>
            <button className="primary-button" onClick={() => setActivePage("Upload data")}>
              <FileUp size={17} /> Upload customer data
            </button>
          </div>

          <div className="demo-notice">
            <AlertTriangle size={17} />
            <span><strong>Demonstration data:</strong> The figures below are illustrative and are not connected to live customer records yet.</span>
          </div>

          <div className="section-heading">
            <div>
              <h2>Performance snapshot</h2>
              <p>Illustrative workspace metrics</p>
            </div>
            <button className="period-select">Last 30 days <ChevronDown size={15} /></button>
          </div>

          <div className="metrics-grid">
            <MetricCard
              title="Total customers"
              value="7,043"
              change="+4.8%"
              direction="up"
              note="compared with previous period"
              icon={<Users size={19} />}
              iconClass="metric-blue"
            />
            <MetricCard
              title="Churn rate"
              value="26.5%"
              change="-2.1%"
              direction="down"
              note="compared with previous period"
              icon={<Activity size={19} />}
              iconClass="metric-teal"
            />
            <MetricCard
              title="High-risk customers"
              value="1,284"
              change="+3.2%"
              direction="up"
              note="compared with previous period"
              icon={<AlertTriangle size={19} />}
              iconClass="metric-orange"
            />
            <MetricCard
              title="Predictions generated"
              value="2,846"
              change="+12.6%"
              direction="up"
              note="compared with previous period"
              icon={<Zap size={19} />}
              iconClass="metric-purple"
            />
          </div>

          <div className="charts-grid">
            <section className="panel trend-panel">
              <div className="panel-heading">
                <div>
                  <h3>Customer activity</h3>
                  <p>Monthly customer volume and churned customers</p>
                </div>
                <button className="chart-menu">Last 7 months <ChevronDown size={14} /></button>
              </div>
              <div className="chart-legend">
                <span><i className="legend-dot customers-dot" /> Customer volume</span>
                <span><i className="legend-dot churn-dot" /> Churned customers</span>
              </div>
              <div className="trend-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={trendData} margin={{ top: 10, right: 8, left: -18, bottom: 0 }}>
                    <defs>
                      <linearGradient id="customerGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#2563eb" stopOpacity={0.16} />
                        <stop offset="100%" stopColor="#2563eb" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="churnGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.15} />
                        <stop offset="100%" stopColor="#f59e0b" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#edf0f4" strokeDasharray="4 4" vertical={false} />
                    <XAxis dataKey="month" axisLine={false} tickLine={false} tick={{ fill: "#8a94a6", fontSize: 12 }} dy={10} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fill: "#8a94a6", fontSize: 12 }} />
                    <Tooltip contentStyle={{ borderRadius: 10, border: "1px solid #e9edf2", fontSize: 12 }} />
                    <Area type="monotone" dataKey="customers" name="Customers" stroke="#2563eb" strokeWidth={2.5} fill="url(#customerGradient)" />
                    <Area type="monotone" dataKey="churned" name="Churned" stroke="#f59e0b" strokeWidth={2.5} fill="url(#churnGradient)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="panel segment-panel">
              <div className="panel-heading">
                <div>
                  <h3>Customer risk segments</h3>
                  <p>Distribution by predicted churn risk</p>
                </div>
                <button className="more-button" aria-label="More options">···</button>
              </div>
              <div className="segment-chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={segmentData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={67} outerRadius={91} paddingAngle={4} stroke="none">
                      {segmentData.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
                    </Pie>
                    <Tooltip formatter={(value) => `${value}%`} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="donut-center">
                  <strong>7,043</strong>
                  <span>Customers</span>
                </div>
              </div>
              <div className="segment-legend">
                {segmentData.map((segment) => (
                  <div className="segment-row" key={segment.name}>
                    <span><i style={{ background: segment.color }} />{segment.name}</span>
                    <strong>{segment.value}%</strong>
                  </div>
                ))}
              </div>
            </section>
          </div>

          <section className="panel predictions-panel">
            <div className="panel-heading predictions-heading">
              <div>
                <h3>Recent predictions</h3>
                <p>Latest customer churn risk assessments</p>
              </div>
              <button className="text-button" onClick={() => setActivePage("History")}>View history <ArrowUpRight size={15} /></button>
            </div>
            <div className="table-scroll">
              <table className="predictions-table">
                <thead>
                  <tr>
                    <th>Customer ID</th>
                    <th>Contract type</th>
                    <th>Churn probability</th>
                    <th>Risk level</th>
                    <th>Prediction date</th>
                  </tr>
                </thead>
                <tbody>
                  {recentPredictions.map((item) => (
                    <tr key={item.id}>
                      <td className="customer-id">{item.id}</td>
                      <td>{item.contract}</td>
                      <td>
                        <div className="probability-cell">
                          <div className="probability-track"><span style={{ width: `${item.probability * 100}%` }} /></div>
                          <strong>{Math.round(item.probability * 100)}%</strong>
                        </div>
                      </td>
                      <td><span className={`risk-badge risk-${item.risk.toLowerCase()}`}>{item.risk}</span></td>
                      <td className="date-cell">{item.date}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="table-footer">
              <span>Showing 4 illustrative records</span>
              <button onClick={() => setActivePage("Predictions")}>Explore predictions <ArrowUpRight size={14} /></button>
            </div>
          </section>

          <footer className="footer">
            <span>© 2026 ChurnOps. Customer intelligence platform.</span>
            <span><i className="footer-dot" /> Demo mode · API integration pending</span>
          </footer>
        </div>
      </section>
    </main>
  );
}

function MetricCard({
  title,
  value,
  change,
  direction,
  note,
  icon,
  iconClass,
}: {
  title: string;
  value: string;
  change: string;
  direction: "up" | "down";
  note: string;
  icon: React.ReactNode;
  iconClass: string;
}) {
  return (
    <section className="metric-card">
      <div className="metric-top">
        <span>{title}</span>
        <div className={`metric-icon ${iconClass}`}>{icon}</div>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-bottom">
        <span className={`metric-change ${direction === "down" ? "change-positive" : "change-neutral"}`}>
          {direction === "up" ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
          {change}
        </span>
        <span className="metric-note">{note}</span>
      </div>
    </section>
  );
}
