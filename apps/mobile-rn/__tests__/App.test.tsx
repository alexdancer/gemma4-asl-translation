/**
 * @format
 */

import React from 'react';
import {create} from 'react-test-renderer';

jest.mock('@react-native-documents/picker', () => ({
  pick: jest.fn(),
  isErrorWithCode: jest.fn(() => false),
  types: {video: 'public.movie'},
}));

import App from '../App';

test('renders cloud translation shell', () => {
  const tree = create(<App />);
  expect(tree).toBeTruthy();
  tree.unmount();
});
