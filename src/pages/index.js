import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import clsx from 'clsx';

import styles from './styles.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroText}>
            <h1 className="hero__title">{siteConfig.title}</h1>
            <p className="hero__subtitle">{siteConfig.tagline}</p>
            <div className={styles.heroButtons}>
              <a
                className="button button--secondary button--lg"
                href="/docs/module-1-ros/intro"
              >
                Start Learning 🚀
              </a>
              <a
                className="button button--outline button--lg"
                href="/docs/complete-documentation"
              >
                View Complete Documentation
              </a>
            </div>
          </div>
          <div className={styles.heroImage}>
            <div className={styles.robotAnimation}>
              <div className={styles.robotArm}></div>
              <div className={styles.robotBody}></div>
              <div className={styles.robotHead}></div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function HomepageContent() {
  return (
    <div className={styles.homepageContent}>
      <section className={styles.features}>
        <div className="container">
          <div className="row">
            <div className={clsx('col col--4', styles.featureCard)}>
              <div className={styles.featureIcon}>
                <div className={styles.iconROS}></div>
              </div>
              <h2>ROS 2 Fundamentals</h2>
              <p>Master the Robot Operating System 2, the backbone of modern robotics communication and control.</p>
            </div>
            <div className={clsx('col col--4', styles.featureCard)}>
              <div className={styles.featureIcon}>
                <div className={styles.iconSimulation}></div>
              </div>
              <h2>Digital Twins</h2>
              <p>Build and simulate realistic robot models using Gazebo and Unity for safe development and testing.</p>
            </div>
            <div className={clsx('col col--4', styles.featureCard)}>
              <div className={styles.featureIcon}>
                <div className={styles.iconAI}></div>
              </div>
              <h2>AI Integration</h2>
              <p>Implement NVIDIA Isaac technologies for GPU-accelerated perception and intelligent behavior.</p>
            </div>
          </div>
          <div className="row" style={{marginTop: '2rem'}}>
            <div className={clsx('col col--4', styles.featureCard)}>
              <div className={styles.featureIcon}>
                <div className={styles.iconVLA}></div>
              </div>
              <h2>Vision-Language-Action</h2>
              <p>Combine perception, language understanding, and physical action in unified robotic systems.</p>
            </div>
            <div className={clsx('col col--4', styles.featureCard)}>
              <div className={styles.featureIcon}>
                <div className={styles.iconTransfer}></div>
              </div>
              <h2>Simulation to Reality</h2>
              <p>Transfer learning from simulation to real-world robots with domain randomization.</p>
            </div>
            <div className={clsx('col col--4', styles.featureCard)}>
              <div className={styles.featureIcon}>
                <div className={styles.iconProject}></div>
              </div>
              <h2>Capstone Project</h2>
              <p>Integrate all concepts in a complete autonomous humanoid robot system.</p>
            </div>
          </div>
        </div>
      </section>

      <section className={styles.about}>
        <div className="container">
          <div className="row">
            <div className="col col--12">
              <h2>Physical AI & Humanoid Robotics</h2>
              <p>
                This comprehensive textbook provides a complete educational framework covering modern robotics technologies,
                from fundamental ROS 2 concepts to advanced Vision-Language-Action systems powered by NVIDIA Isaac.
              </p>
              <p>
                Each module builds upon previous concepts while introducing new technologies and capabilities,
                creating a cohesive learning experience that leads to the ultimate goal of autonomous humanoid robotics.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className={styles.modules}>
        <div className="container">
          <div className="row">
            <div className="col col--12">
              <h2>Learning Path</h2>
              <div className={styles.moduleGrid}>
                <div className={clsx(styles.moduleCard, styles.moduleCard1)}>
                  <div className={styles.moduleBadge}>Module 1</div>
                  <h3>The Robotic Nervous System (ROS 2)</h3>
                  <p>Establishes the foundational concepts of Robot Operating System 2 (ROS 2),
                     the middleware that enables communication between different robotic components.</p>
                  <div className={styles.moduleStats}>
                    <span>9 chapters</span>
                    <span>3 projects</span>
                  </div>
                </div>
                <div className={clsx(styles.moduleCard, styles.moduleCard2)}>
                  <div className={styles.moduleBadge}>Module 2</div>
                  <h3>The Digital Twin (Gazebo & Unity)</h3>
                  <p>Focuses on creating digital twins using Gazebo physics simulation and Unity HDRP
                     for realistic visualization. Digital twins enable safe testing and development of robotic systems.</p>
                  <div className={styles.moduleStats}>
                    <span>7 chapters</span>
                    <span>2 projects</span>
                  </div>
                </div>
                <div className={clsx(styles.moduleCard, styles.moduleCard3)}>
                  <div className={styles.moduleBadge}>Module 3</div>
                  <h3>The AI-Robot Brain (NVIDIA Isaac™)</h3>
                  <p>Introduces NVIDIA Isaac technologies, including Isaac Sim for photorealistic simulation
                     and Isaac ROS for GPU-accelerated perception. This module focuses on the AI components
                     that enable intelligent robotic behavior.</p>
                  <div className={styles.moduleStats}>
                    <span>8 chapters</span>
                    <span>3 projects</span>
                  </div>
                </div>
                <div className={clsx(styles.moduleCard, styles.moduleCard4)}>
                  <div className={styles.moduleBadge}>Module 4</div>
                  <h3>Vision-Language-Action (VLA)</h3>
                  <p>Implements Vision-Language-Action systems that integrate perception, language understanding,
                     and physical action in unified frameworks. This represents the cutting edge of embodied AI.</p>
                  <div className={styles.moduleStats}>
                    <span>6 chapters</span>
                    <span>2 projects</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className={styles.cta}>
        <div className="container">
          <div className="row">
            <div className="col col--12 text--center">
              <h2>Ready to Build the Future of Robotics?</h2>
              <p>Join thousands of students and professionals learning the cutting-edge technologies that power humanoid robots.</p>
              <a
                className="button button--primary button--lg"
                href="/docs/module-1-ros/intro"
              >
                Begin Your Journey
              </a>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="A comprehensive guide to Physical AI and Humanoid Robotics: ROS 2, Digital Twins, NVIDIA Isaac, and Vision-Language-Action Systems">
      <HomepageHeader />
      <main>
        <HomepageContent />
      </main>
    </Layout>
  );
}