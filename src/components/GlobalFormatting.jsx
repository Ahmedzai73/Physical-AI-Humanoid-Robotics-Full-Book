import React from 'react';
import clsx from 'clsx';
import styles from './GlobalFormatting.module.css';

// Global formatting components for the textbook
const FeatureList = [
  {
    title: 'Practical Examples',
    Svg: require('../../static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Real-world examples that demonstrate concepts in practice.
      </>
    ),
  },
  {
    title: 'NVIDIA Isaac Ecosystem',
    Svg: require('../../static/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Integration with NVIDIA Isaac tools and frameworks.
      </>
    ),
  },
  {
    title: 'Hands-on Projects',
    Svg: require('../../static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Projects that let you apply what you've learned.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}