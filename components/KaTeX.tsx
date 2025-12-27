import React, { useMemo } from 'react';
import katex from 'katex';

interface KaTeXProps {
  math: string;
  block?: boolean;
  className?: string;
}

const KaTeX: React.FC<KaTeXProps> = ({ math, block = false, className = "" }) => {
  const html = useMemo(() => {
    if (!math) return "";
    try {
      return katex.renderToString(math, {
        displayMode: block,
        throwOnError: false,
        trust: true,
        strict: false
      });
    } catch (error) {
      console.warn("KaTeX renderToString failure:", error);
      return math;
    }
  }, [math, block]);

  return (
    <span 
      className={`${className} inline-block`}
      dangerouslySetInnerHTML={{ __html: html }} 
    />
  );
};

export default KaTeX;